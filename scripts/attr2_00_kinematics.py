"""V2 attribution - stage 0: kinematics tables, bout structure, and the
kinematics-emission GATE test.

Idea (Warren & Neunuebel, PNAS 2018; Sangiamo 2020): the CALLER's own movement
predicts USV emission - callers move faster, accelerate, and turn while
vocalizing. The alone phase provides ground truth (every call is the
resident's), so we can measure that coupling on this dataset and, if it holds,
use it in the partner phase to compare the two animals' kinematics at each
call. This cue needs no stim IDs, no voice fingerprint, and no level model.

This stage:
1. Builds a 30 fps kinematic table per session for both mice (speed, accel,
   turn rate, and inter-animal features in the partner phase).
2. Segments calls into bouts from the inter-call-interval distribution.
3. GATE: leave-one-animal-out logistic AUC for "call frame vs control frame"
   from the caller's kinematics in the ALONE phase. If this is ~0.5 the cue
   is dead and v2 must rely on other cues.

Outputs: attribution/v2_kinematics.npz (per-session frame tables),
attribution/v2_bouts.csv, attribution/v2_kin_gate.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A
import attr2_core as C2

SMOOTH_SD_S = 0.10          # Gaussian smoothing of position traces
CONTROL_EXCLUDE_S = 1.0     # control frames stay >=1 s away from any call
BOUT_GAP_S = 0.5            # bout break threshold (checked against ICI dist)

KIN_FEATS = ["speed", "accel", "turn"]


def session_kinematics(animal: str) -> pd.DataFrame:
    """30 fps table: t, per-mouse speed/accel/turn (px/s units), dist, approach."""
    tr = A.load_tracking(animal)
    t = tr["t"].to_numpy()
    out = {"t": t}
    sd = SMOOTH_SD_S * A.FPS
    for m in ("m1", "m2"):
        x = tr[f"{m}_x"].to_numpy().copy()
        y = tr[f"{m}_y"].to_numpy().copy()
        ok = np.isfinite(x)
        if ok.sum() > 10:
            x = np.interp(t, t[ok], x[ok])
            y = np.interp(t, t[ok], y[ok])
            xs = gaussian_filter1d(x, sd)
            ys = gaussian_filter1d(y, sd)
            vx = np.gradient(xs, t)
            vy = np.gradient(ys, t)
            speed = np.hypot(vx, vy)
            accel = np.gradient(speed, t)
            heading = np.unwrap(np.arctan2(vy, vx))
            turn = np.abs(np.gradient(heading, t))
            turn = np.clip(turn, 0, 20)
            # invalidate long tracking gaps
            gap = np.ones_like(t, bool)
            gap[ok] = False
            bad = gaussian_filter1d(gap.astype(float), sd) > 0.5
            for arr in (speed, accel, turn):
                arr[bad] = np.nan
        else:
            xs = ys = speed = accel = turn = np.full_like(t, np.nan)
        out[f"{m}_x"] = xs
        out[f"{m}_y"] = ys
        out[f"{m}_speed"] = speed
        out[f"{m}_accel"] = accel
        out[f"{m}_turn"] = turn
    d = np.hypot(out["m1_x"] - out["m2_x"], out["m1_y"] - out["m2_y"])
    d[t < A.PARTNER_INTRO_S] = np.nan
    out["dist"] = d
    ddist = np.gradient(gaussian_filter1d(np.nan_to_num(d, nan=0.0), sd), t)
    ddist[~np.isfinite(d)] = np.nan
    out["ddist"] = ddist   # negative = animals closing distance
    return pd.DataFrame(out)


def zscore_per_session(v: np.ndarray) -> np.ndarray:
    mu, sd = np.nanmean(v), np.nanstd(v)
    return (v - mu) / (sd + 1e-9)


def build_all():
    calls = A.load_calls()
    kin = {}
    for animal in A.GENOTYPE_MAP:
        kin[animal] = session_kinematics(animal)
    return calls, kin


def bout_segment(calls: pd.DataFrame) -> pd.DataFrame:
    """Assign bout ids per session using the BOUT_GAP_S threshold."""
    rows = []
    for animal, g in calls.groupby("animal_id"):
        g = g.sort_values("start_s")
        gaps = g["start_s"].diff().fillna(np.inf)
        bout = (gaps > BOUT_GAP_S).cumsum()
        for cid, b in zip(g["call_id"], bout):
            rows.append({"call_id": cid, "bout_id": f"{animal}_b{int(b)}"})
    return pd.DataFrame(rows)


def kin_features_at(kin: pd.DataFrame, times: np.ndarray, mouse: str,
                    z: dict | None = None) -> np.ndarray:
    """Feature matrix [speed, accel, turn] for one mouse at given times,
    z-scored with the session-wide stats passed in `z` (or computed here)."""
    t = kin["t"].to_numpy()
    cols = []
    for f in KIN_FEATS:
        v = kin[f"{mouse}_{f}"].to_numpy()
        ok = np.isfinite(v)
        vi = np.interp(times, t[ok], v[ok]) if ok.sum() > 10 else np.full_like(times, np.nan)
        if z is not None:
            mu, sd = z[f"{mouse}_{f}"]
        else:
            mu, sd = np.nanmean(v), np.nanstd(v)
        cols.append((vi - mu) / (sd + 1e-9))
    return np.stack(cols, axis=1)


def session_zstats(kin: pd.DataFrame, phase: str) -> dict:
    """Per-mouse feature stats within a phase (alone/partner) for z-scoring."""
    t = kin["t"].to_numpy()
    m = t < A.PARTNER_INTRO_S if phase == "alone" else t >= A.PARTNER_INTRO_S
    out = {}
    for mouse in ("m1", "m2"):
        for f in KIN_FEATS:
            v = kin[f"{mouse}_{f}"].to_numpy()[m]
            out[f"{mouse}_{f}"] = (float(np.nanmean(v)), float(np.nanstd(v)))
    return out


def gate_test(calls: pd.DataFrame, kin: dict) -> dict:
    """LOAO logistic AUC: alone-phase call frames vs control frames, resident
    kinematics only. This is the go/no-go for the kinematic cue."""
    rng = np.random.default_rng(A.RNG_SEED)
    X_all, y_all, an_all = [], [], []
    for animal in A.GENOTYPE_MAP:
        k = kin[animal]
        g = calls[(calls["animal_id"] == animal) & (calls["phase"] == "alone")]
        if not len(g):
            continue
        ct = ((g["start_s"] + g["end_s"]) / 2).to_numpy()
        Xc = C2.kin_feature_matrix(k, "m1", ct, "alone")
        # controls: alone-phase times >=1 s from any call
        tt = np.arange(2.0, A.PARTNER_INTRO_S - 2.0, 1 / 3)
        far = np.ones(len(tt), bool)
        for s in ct:
            far &= np.abs(tt - s) > CONTROL_EXCLUDE_S
        ctrl = rng.choice(tt[far], size=min(far.sum(), 10 * len(g)), replace=False)
        Xn = C2.kin_feature_matrix(k, "m1", ctrl, "alone")
        X_all.append(np.vstack([Xc, Xn]))
        y_all.append(np.concatenate([np.ones(len(Xc)), np.zeros(len(Xn))]))
        an_all.append(np.repeat(animal, len(Xc) + len(Xn)))
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    an = np.concatenate(an_all)
    ok = np.isfinite(X).all(axis=1)
    X, y, an = X[ok], y[ok], an[ok]

    aucs, coefs = [], []
    for a in np.unique(an):
        tr, te = an != a, an == a
        if len(np.unique(y[te])) < 2 or y[te].sum() < 3:
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
        coefs.append(clf.coef_[0])
    clf_full = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    feat_names = KIN_FEATS + [f"speed@{l:+.1f}s" for l in C2.KIN_LAGS]
    return {
        "loao_auc_mean": float(np.mean(aucs)), "loao_auc_sd": float(np.std(aucs)),
        "loao_n_animals": len(aucs),
        "pooled_coef": dict(zip(feat_names, clf_full.coef_[0].round(3).tolist())),
        "n_call_frames": int(y.sum()), "n_control_frames": int((1 - y).sum()),
        "per_animal_auc": [float(a) for a in aucs],
    }, clf_full


def main():
    out = A.ensure_out()
    calls, kin = build_all()

    bouts = bout_segment(calls)
    bouts.to_csv(out / "v2_bouts.csv", index=False)
    ici = calls.sort_values(["animal_id", "start_s"]).groupby("animal_id")["start_s"].diff().dropna()
    nb = bouts["bout_id"].nunique()
    print(f"{len(calls)} calls -> {nb} bouts (gap>{BOUT_GAP_S}s); "
          f"median ICI within bouts: {ici[ici < BOUT_GAP_S].median()*1000:.0f} ms; "
          f"median bout size: {bouts.groupby('bout_id').size().median():.0f}")

    gate, clf = gate_test(calls, kin)
    print(json.dumps(gate, indent=2))
    with open(out / "v2_kin_gate.json", "w") as fh:
        json.dump(gate, fh, indent=2)

    # persist frame tables + logistic weights for later stages
    np.savez_compressed(
        out / "v2_kinematics.npz",
        animals=np.array(list(A.GENOTYPE_MAP)),
        **{f"kin_{a}": kin[a].to_numpy(dtype=np.float32) for a in A.GENOTYPE_MAP},
        columns=np.array(session_kinematics.__doc__ and list(kin[next(iter(kin))].columns)),
        logit_coef=clf.coef_[0], logit_intercept=clf.intercept_,
    )
    print(f"wrote v2_kinematics.npz, v2_bouts.csv, v2_kin_gate.json")


if __name__ == "__main__":
    main()
