"""Per-call feature extraction for single-mic voice attribution.

For every detected call (existing merged detections; no re-detection, no
pretrained models) this script extracts, directly from the raw 384 kHz WAV:

1. Handcrafted acoustic features from a denoised spectrogram ridge
   (frequency trajectory shape, bandwidth, amplitude statistics, tonality).
2. A fixed-size 64x64 SNR-spectrogram crop, used later to train a small
   from-scratch convolutional autoencoder (attr_20).
3. Tracking joins: both animals' positions/speeds at call time, inter-animal
   distance (mouse1 = resident, mouse2 = partner).
4. Behavior flags at call time from the live_detections stream.

Optionally (--mix-with B) audio from another session's alone phase can be
superimposed before extraction, which the audio-level benchmark uses to test
feature robustness with ground-truth speaker labels.

Outputs: attribution/call_features.csv, attribution/call_specs.npz
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

BEHAVIOR_COLS = [
    "behavior_nose2nose", "behavior_sidebyside", "behavior_sidereside",
    "behavior_nose2anogenital", "behavior_nose2body", "behavior_oriented_toward",
    "behavior_following", "behavior_chasing", "behavior_approach",
    "behavior_withdrawal_from_partner", "behavior_escape",
    "behavior_withdrawal_after_contact", "behavior_fighting",
    "behavior_rearing", "behavior_passive",
]


def spec_snr(x: np.ndarray):
    """Denoised (noise-floor-subtracted) dB spectrogram limited to the USV band."""
    f, t, S = signal.spectrogram(
        x, fs=A.SR, window="hann", nperseg=A.NPERSEG, noverlap=A.NOVERLAP,
        detrend=False, mode="psd",
    )
    band = (f >= A.F_LO) & (f <= A.F_HI)
    S = 10.0 * np.log10(S[band] + 1e-20)
    return f[band], t, S


def extract_call(x: np.ndarray, call_rel_start: float, call_rel_end: float):
    """Features + crop for one call given audio context (call plus >=80 ms pad)."""
    f, t, S = spec_snr(x)
    in_call = (t >= call_rel_start - 0.003) & (t <= call_rel_end + 0.003)
    out_call = ~((t >= call_rel_start - 0.01) & (t <= call_rel_end + 0.01))
    if in_call.sum() < 2 or out_call.sum() < 10:
        return None
    floor = np.median(S[:, out_call], axis=1, keepdims=True)
    snr = S - floor

    C = snr[:, in_call]
    tt = t[in_call]
    peak_idx = np.argmax(C, axis=0)
    peak_snr = C[peak_idx, np.arange(C.shape[1])]
    keep = peak_snr >= A.RIDGE_SNR_DB
    if keep.sum() < 2:
        keep = peak_snr >= np.percentile(peak_snr, 60)
    ridge_f = f[peak_idx[keep]]
    ridge_a = peak_snr[keep]
    ridge_t = tt[keep]
    dur = float(ridge_t[-1] - ridge_t[0] + 1e-9)

    # per-frame spectral spread and entropy around the ridge
    P = np.clip(C[:, keep], 0, None)
    Pn = P / (P.sum(axis=0, keepdims=True) + 1e-12)
    ent = float(np.mean(-np.sum(Pn * np.log2(Pn + 1e-12), axis=0)))
    fgrid = f[:, None]
    mu = np.sum(Pn * fgrid, axis=0)
    spread = np.sqrt(np.sum(Pn * (fgrid - mu[None, :]) ** 2, axis=0))

    df = np.diff(ridge_f)
    slope = 0.0
    curve = 0.0
    if len(ridge_t) >= 3:
        z = np.polyfit(ridge_t - ridge_t[0], ridge_f, 1)
        slope = float(z[0])
        z2 = np.polyfit(ridge_t - ridge_t[0], ridge_f, 2)
        curve = float(z2[0])
    path_len = float(np.sum(np.abs(df)))
    straight = float(abs(ridge_f[-1] - ridge_f[0]))
    # position of the amplitude peak along the call (0=onset, 1=offset)
    peak_pos = float((ridge_t[np.argmax(ridge_a)] - ridge_t[0]) / dur) if dur > 0 else 0.5

    feats = {
        "rf_dur_ms": dur * 1e3,
        "rf_f_mean_khz": float(np.mean(ridge_f)) / 1e3,
        "rf_f_std_khz": float(np.std(ridge_f)) / 1e3,
        "rf_f_min_khz": float(np.min(ridge_f)) / 1e3,
        "rf_f_max_khz": float(np.max(ridge_f)) / 1e3,
        "rf_f_start_khz": float(ridge_f[0]) / 1e3,
        "rf_f_end_khz": float(ridge_f[-1]) / 1e3,
        "rf_slope_khz_ms": slope / 1e6,
        "rf_abs_slope_khz_ms": abs(slope) / 1e6,
        "rf_curve_khz_ms2": curve / 1e9,
        "rf_sinuosity": path_len / (straight + 750.0),
        "rf_jumps": int(np.sum(np.abs(df) > 10_000)),
        "rf_frac_rising": float(np.mean(df > 0)) if len(df) else 0.0,
        "rf_entropy_bits": ent,
        "rf_spread_khz": float(np.mean(spread)) / 1e3,
        "rf_amp_peak_db": float(np.max(ridge_a)),
        "rf_amp_mean_db": float(np.mean(ridge_a)),
        "rf_amp_cv": float(np.std(ridge_a) / (np.mean(ridge_a) + 1e-9)),
        "rf_amp_peak_pos": peak_pos,
        "rf_tonality": float(np.mean(np.max(Pn, axis=0))),
        "rf_n_frames": int(keep.sum()),
    }

    # fixed-size crop: full USV band x call span (+-3 ms), SNR clipped at 0
    crop = np.clip(snr[:, in_call], 0, None)
    if cv2 is not None:
        crop = cv2.resize(crop, (A.CROP_SIZE, A.CROP_SIZE), interpolation=cv2.INTER_AREA)
    else:
        ty = np.linspace(0, crop.shape[0] - 1, A.CROP_SIZE).astype(int)
        tx = np.linspace(0, crop.shape[1] - 1, A.CROP_SIZE).astype(int)
        crop = crop[np.ix_(ty, tx)]
    peak = float(crop.max())
    feats["crop_peak_db"] = peak
    crop = (crop / (peak + 1e-9)).astype(np.float16)
    return feats, crop


def load_behavior(animal: str) -> pd.DataFrame | None:
    """Per-frame dyadic behavior flags (max over the two per-mouse rows)."""
    try:
        cols = ["frame_id"] + BEHAVIOR_COLS
        b = pd.read_csv(A.live_detections_path(animal), usecols=cols)
    except Exception:
        return None
    return b.groupby("frame_id", as_index=True).max()


def process_animal(animal: str, calls: pd.DataFrame, mix_with: str | None = None,
                   mix_gain: float = 1.0):
    """Extract features for all calls of one session (optionally with mixed-in audio)."""
    rows, crops, ids = [], [], []
    wav = A.wav_path(animal)
    info = sf.info(str(wav))
    other = None
    if mix_with is not None:
        other_info = sf.info(str(A.wav_path(mix_with)))

    track = A.load_tracking(animal)
    behav = load_behavior(animal)
    sub = calls[calls["animal_id"] == animal]
    mids = ((sub["start_s"] + sub["end_s"]) / 2).to_numpy()
    tr = A.interp_track(track, mids)

    for i, (_, c) in enumerate(sub.iterrows()):
        pad = 0.1
        s0 = max(0.0, c["start_s"] - pad)
        s1 = min(info.frames / A.SR, c["end_s"] + pad)
        start_fr = int(s0 * A.SR)
        n_fr = int((s1 - s0) * A.SR)
        x, _ = sf.read(str(wav), start=start_fr, frames=n_fr, dtype="float64", always_2d=False)
        if mix_with is not None and start_fr + n_fr < other_info.frames:
            y, _ = sf.read(str(A.wav_path(mix_with)), start=start_fr, frames=n_fr,
                           dtype="float64", always_2d=False)
            x = x + mix_gain * y[: len(x)]
        got = extract_call(x, c["start_s"] - s0, c["end_s"] - s0)
        if got is None:
            continue
        feats, crop = got
        feats.update({
            "call_id": c["call_id"], "animal_id": animal,
            "genotype": c["genotype"], "phase": c["phase"],
            "start_s": c["start_s"], "end_s": c["end_s"],
            "csv_avg_intensity": c["avg_intensity"], "csv_bg_intensity": c["bg_intensity"],
            "m1_x": tr["m1_x"][i], "m1_y": tr["m1_y"][i],
            "m2_x": tr["m2_x"][i], "m2_y": tr["m2_y"][i],
            "m1_speed": tr["m1_speed"][i], "m2_speed": tr["m2_speed"][i],
            "dist12": tr["dist12"][i],
            "stim_id": A.STIM_MAP.get(animal) or "",
        })
        if behav is not None:
            frame = int(round(((c["start_s"] + c["end_s"]) / 2) * A.FPS))
            if frame in behav.index:
                for bc in BEHAVIOR_COLS:
                    feats[bc] = int(behav.loc[frame, bc])
        rows.append(feats)
        crops.append(crop)
        ids.append(c["call_id"])
    return rows, crops, ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-prefix", default="call", help="output file prefix")
    ap.add_argument("--animals", nargs="*", default=None)
    args = ap.parse_args()

    out = A.ensure_out()
    calls = A.load_calls()
    animals = args.animals or sorted(calls["animal_id"].unique())

    all_rows, all_crops, all_ids = [], [], []
    for k, animal in enumerate(animals, 1):
        rows, crops, ids = process_animal(animal, calls)
        all_rows += rows
        all_crops += crops
        all_ids += ids
        print(f"[{k}/{len(animals)}] {animal}: {len(rows)} calls", flush=True)

    feats = pd.DataFrame(all_rows)
    feats.to_csv(out / f"{args.out_prefix}_features.csv", index=False)
    np.savez_compressed(
        out / f"{args.out_prefix}_specs.npz",
        specs=np.stack(all_crops).astype(np.float16),
        call_id=np.array(all_ids),
    )
    print(f"wrote {len(feats)} calls -> {out}")


if __name__ == "__main__":
    main()
