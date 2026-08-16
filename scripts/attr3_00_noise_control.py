"""Are the detected "calls" real USVs, or mechanical noise from moving mice?

The counter-hypothesis: detections are false positives caused by locomotion
(footfalls, bedding rustle, scratching, cage contact). If true, the whole
movement-based attribution in v2 would be circular, and the WT/HET difference
could simply reflect WT dyads moving more.

The decisive test exploits the detector's own blindness: it only examined
45-125 kHz (parameters.yml lower_frequency_cutoff=45000). Energy BELOW that
band was never used for selection, so it is a free, unbiased measurement.
  - A true USV is a narrowband whistle: loud in 45-125 kHz, silent in 5-35 kHz.
  - A mechanical transient is broadband: it MUST also appear in 5-35 kHz.

Pass A (streaming, all sessions): band-energy envelopes at 750 Hz for the
  sub-USV band (5-35 kHz) and the USV band (45-125 kHz). Yields per-call
  low-band SNR, envelope-vs-locomotion correlations, and an independent
  mechanical-transient detector.
Pass B (targeted): full-band spectrogram crops + shape features for detected
  calls, movement-matched control times, and detected mechanical transients.

Outputs (attribution/):
  noise_envelopes.npz, noise_call_measures.csv, noise_control_measures.csv,
  noise_crops.npz, noise_session_stats.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A
import attr2_core as C2

ENV_NPERSEG = 1024
ENV_NOVERLAP = 512
ENV_RATE = A.SR / (ENV_NPERSEG - ENV_NOVERLAP)      # 750 Hz
BAND_LOW = (5_000.0, 35_000.0)      # below the detector's window
BAND_USV = (45_000.0, 125_000.0)    # the detector's window
CHUNK_S = 30.0
BASELINE_WIN_S = 2.0
TRANSIENT_THR_DB = 6.0
TRANSIENT_MIN_SEP_S = 0.05
CTRL_EXCLUDE_S = 1.0
N_CROP_PER_GROUP = 400
CROP_PAD_S = 0.05
SEED = A.RNG_SEED


# --------------------------------------------------------------- pass A
def stream_band_envelopes(path):
    """750 Hz energy envelopes (dB) for the sub-USV and USV bands."""
    info = sf.info(str(path))
    step = int(CHUNK_S * A.SR)
    lows, usvs = [], []
    pos = 0
    while pos < info.frames:
        n = min(step, info.frames - pos)
        if n < ENV_NPERSEG:            # trailing remainder shorter than one FFT
            break
        x, _ = sf.read(str(path), start=pos, frames=n, dtype="float32")
        f, _, S = signal.spectrogram(
            x, fs=A.SR, window="hann", nperseg=ENV_NPERSEG,
            noverlap=ENV_NOVERLAP, mode="psd")
        lo = (f >= BAND_LOW[0]) & (f <= BAND_LOW[1])
        hi = (f >= BAND_USV[0]) & (f <= BAND_USV[1])
        lows.append(10 * np.log10(S[lo].mean(axis=0) + 1e-20))
        usvs.append(10 * np.log10(S[hi].mean(axis=0) + 1e-20))
        pos += n
    low = np.concatenate(lows).astype(np.float32)
    usv = np.concatenate(usvs).astype(np.float32)
    t = np.arange(len(low)) / ENV_RATE
    return t, low, usv


def baseline_subtract(env: np.ndarray) -> np.ndarray:
    """Local baseline removal (running median over BASELINE_WIN_S)."""
    w = int(BASELINE_WIN_S * ENV_RATE) | 1
    return env - median_filter(env, size=w, mode="nearest")


def sample_at(t_env, env, times):
    idx = np.clip(np.round(np.asarray(times) * ENV_RATE).astype(int),
                  0, len(env) - 1)
    return env[idx]


def detect_transients(t_env, low_bs):
    """Mechanical-event detector run ONLY on the sub-USV band."""
    peaks, _ = signal.find_peaks(
        low_bs, height=TRANSIENT_THR_DB,
        distance=max(1, int(TRANSIENT_MIN_SEP_S * ENV_RATE)))
    return t_env[peaks], low_bs[peaks]


# --------------------------------------------------------------- controls
def movement_trace(kin: pd.DataFrame):
    """Mechanical-noise proxy: total arena movement (both animals' speed)."""
    t = kin["t"].to_numpy()
    s1 = np.nan_to_num(kin["m1_speed"].to_numpy())
    s2 = np.nan_to_num(kin["m2_speed"].to_numpy())
    tot = np.where(t < A.PARTNER_INTRO_S, s1, s1 + s2)
    return t, tot


def matched_controls(kin, call_times, rng, tol=0.15):
    """For each call, a control time in the same phase with matched movement
    and no call within +-CTRL_EXCLUDE_S."""
    t, mv = movement_trace(kin)
    grid = np.arange(2.0, A.SESSION_END_S - 2.0, 0.1)
    mv_grid = np.interp(grid, t, mv)
    call_times = np.asarray(call_times)
    far = np.ones(len(grid), bool)
    for c in call_times:
        far &= np.abs(grid - c) > CTRL_EXCLUDE_S
    out = []
    for c in call_times:
        same_phase = ((grid < A.PARTNER_INTRO_S) == (c < A.PARTNER_INTRO_S))
        target = np.interp(c, t, mv)
        for factor in (1.0, 2.0, 4.0, np.inf):
            band = np.abs(mv_grid - target) <= tol * max(target, 1.0) * factor
            ok = far & same_phase & band
            if ok.sum():
                out.append(rng.choice(grid[ok]))
                break
        else:
            out.append(np.nan)
    return np.array(out)


# --------------------------------------------------------------- pass B
def full_band_crop(path, t_center, dur_s):
    """Full-spectrum SNR crop (0-190 kHz) around an event, plus shape stats."""
    info = sf.info(str(path))
    half = max(dur_s, 0.004) / 2 + CROP_PAD_S
    s0 = max(0.0, t_center - half)
    n = int(min(2 * half, info.frames / A.SR - s0) * A.SR)
    if n < 4096:
        return None
    x, _ = sf.read(str(path), start=int(s0 * A.SR), frames=n, dtype="float32")
    f, t, S = signal.spectrogram(x, fs=A.SR, window="hann", nperseg=512,
                                 noverlap=384, mode="psd")
    S = 10 * np.log10(S + 1e-20)
    ev = (t >= t_center - s0 - dur_s / 2 - 0.002) & \
         (t <= t_center - s0 + dur_s / 2 + 0.002)
    ctx = ~((t >= t_center - s0 - dur_s / 2 - 0.01) &
            (t <= t_center - s0 + dur_s / 2 + 0.01))
    if ev.sum() < 2 or ctx.sum() < 8:
        return None
    snr = S - np.median(S[:, ctx], axis=1, keepdims=True)
    band = (f >= 2_000) & (f <= 190_000)
    crop = np.clip(snr[band][:, ev], 0, None)
    fb = f[band]
    lo = (fb >= BAND_LOW[0]) & (fb <= BAND_LOW[1])
    hi = (fb >= BAND_USV[0]) & (fb <= BAND_USV[1])
    C = crop
    stats = {
        "peak_low_db": float(C[lo].max()) if lo.sum() else np.nan,
        "peak_usv_db": float(C[hi].max()) if hi.sum() else np.nan,
    }
    # tonality / entropy inside the USV band
    P = np.clip(C[hi], 0, None)
    Pn = P / (P.sum(axis=0, keepdims=True) + 1e-12)
    stats["tonality"] = float(np.mean(np.max(Pn, axis=0)))
    stats["entropy_bits"] = float(np.mean(-np.sum(Pn * np.log2(Pn + 1e-12), axis=0)))
    # how much of the total event energy sits in the USV band vs below it
    tot = C[hi].sum() + C[lo].sum() + 1e-9
    stats["usv_energy_frac"] = float(C[hi].sum() / tot)
    # resize crop for display (freq x time -> 96 x 64)
    yi = np.linspace(0, C.shape[0] - 1, 96).astype(int)
    xi = np.linspace(0, C.shape[1] - 1, 64).astype(int)
    disp = C[np.ix_(yi, xi)]
    peak = disp.max() + 1e-9
    return stats, (disp / peak).astype(np.float16)


def main():
    out = A.ensure_out()
    rng = np.random.default_rng(SEED)
    calls = A.load_calls()
    kin, _, _ = C2.load_kinematics()

    call_rows, ctrl_rows, sess_rows = [], [], []
    crops = {"call": [], "control": [], "mech": []}
    env_store = {}

    for i, animal in enumerate(A.GENOTYPE_MAP, 1):
        wav = A.wav_path(animal)
        t_env, low, usv = stream_band_envelopes(wav)
        low_bs = baseline_subtract(low)
        usv_bs = baseline_subtract(usv)
        env_store[f"low_{animal}"] = low_bs
        env_store[f"usv_{animal}"] = usv_bs

        g = calls[calls["animal_id"] == animal].copy()
        ct = ((g["start_s"] + g["end_s"]) / 2).to_numpy()
        dur = (g["end_s"] - g["start_s"]).to_numpy()
        cc = matched_controls(kin[animal], ct, rng)

        mech_t, mech_amp = detect_transients(t_env, low_bs)
        k = kin[animal]
        t_mv, mv = movement_trace(k)
        mv_env = np.interp(t_env, t_mv, mv)
        good = np.isfinite(mv_env)
        r_low = float(np.corrcoef(low_bs[good], mv_env[good])[0, 1])
        r_usv = float(np.corrcoef(usv_bs[good], mv_env[good])[0, 1])

        # per-call and per-control measures from the envelopes
        for cid, tc, d, tk in zip(g["call_id"], ct, dur, cc):
            call_rows.append({
                "call_id": cid, "animal_id": animal,
                "genotype": A.GENOTYPE_MAP[animal],
                "phase": "alone" if tc < A.PARTNER_INTRO_S else "partner",
                "t": tc, "dur_s": d,
                "low_snr_db": float(sample_at(t_env, low_bs, [tc])[0]),
                "usv_snr_db": float(sample_at(t_env, usv_bs, [tc])[0]),
                "speed_m1": float(np.interp(tc, k["t"], np.nan_to_num(k["m1_speed"]))),
                "mech_within_10ms": bool(
                    len(mech_t) and np.min(np.abs(mech_t - tc)) < 0.010),
            })
        ok_ctrl = np.isfinite(cc)
        for tk in cc[ok_ctrl]:
            ctrl_rows.append({
                "animal_id": animal, "genotype": A.GENOTYPE_MAP[animal],
                "phase": "alone" if tk < A.PARTNER_INTRO_S else "partner",
                "t": tk,
                "low_snr_db": float(sample_at(t_env, low_bs, [tk])[0]),
                "usv_snr_db": float(sample_at(t_env, usv_bs, [tk])[0]),
            })

        dist_px = float(np.trapezoid(np.nan_to_num(k["m1_speed"]), k["t"]))
        alone_m = k["t"].to_numpy() < A.PARTNER_INTRO_S
        sess_rows.append({
            "animal_id": animal, "genotype": A.GENOTYPE_MAP[animal],
            "n_calls": len(g),
            "n_mech_events": int(len(mech_t)),
            "mech_rate_per_min": len(mech_t) / (A.SESSION_END_S / 60),
            "corr_lowband_movement": r_low,
            "corr_usvband_movement": r_usv,
            "dist_px_total": dist_px,
            "dist_px_alone": float(np.trapezoid(
                np.nan_to_num(k["m1_speed"])[alone_m], k["t"][alone_m])),
            "dist_px_partner": float(np.trapezoid(
                np.nan_to_num(k["m1_speed"])[~alone_m], k["t"][~alone_m])),
        })

        # targeted crops: a few calls, controls, and mechanical events
        per = max(1, N_CROP_PER_GROUP // len(A.GENOTYPE_MAP))
        for tag, times, durs in (
                ("call", ct, dur),
                ("control", cc[ok_ctrl], rng.choice(dur, ok_ctrl.sum())),
                ("mech", mech_t, np.full(len(mech_t), 0.01))):
            if not len(times):
                continue
            pick = rng.choice(len(times), min(per, len(times)), replace=False)
            for j in pick:
                got = full_band_crop(wav, times[j], float(durs[j]))
                if got is None:
                    continue
                st, disp = got
                st.update({"tag": tag, "animal_id": animal, "t": float(times[j])})
                crops[tag].append((st, disp))
        print(f"[{i}/24] {animal}: {len(g)} calls, {len(mech_t)} mech events, "
              f"r(low,move)={r_low:.2f} r(usv,move)={r_usv:.2f}", flush=True)

    pd.DataFrame(call_rows).to_csv(out / "noise_call_measures.csv", index=False)
    pd.DataFrame(ctrl_rows).to_csv(out / "noise_control_measures.csv", index=False)
    pd.DataFrame(sess_rows).to_csv(out / "noise_session_stats.csv", index=False)
    np.savez_compressed(out / "noise_envelopes.npz", rate=ENV_RATE, **env_store)

    shape_rows, arrs, tags = [], [], []
    for tag, items in crops.items():
        for st, disp in items:
            shape_rows.append(st)
            arrs.append(disp)
            tags.append(tag)
    pd.DataFrame(shape_rows).to_csv(out / "noise_crop_stats.csv", index=False)
    np.savez_compressed(out / "noise_crops.npz",
                        crops=np.stack(arrs).astype(np.float16),
                        tag=np.array(tags))
    print("wrote noise control tables + crops")


if __name__ == "__main__":
    main()
