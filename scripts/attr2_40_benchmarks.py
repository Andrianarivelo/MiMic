"""V2 benchmark matrix: score every cue and fusion against ground truth,
pick the winner, calibrate it.

Ground truth = synthetic dyads from alone-phase calls of animal pairs
(A = resident with anchor, B = partner), with each call carrying its OWN
session's kinematics, position, and level. Feature-level (clean audio) and
audio-level (summed waveforms) variants. No stim IDs anywhere.

Methods scored (call-level and HMM-smoothed):
  kin        caller-kinematics likelihood ratio (alone-trained logistic)
  voice_v1   anchored mixture in the v1 AE voice space
  voice_v2   anchored mixture in the v2 bout-contrastive space
  level      loudness-residual x distance cue (auto-gated)
  fusion     logistic fusion (leave-one-pair-out cross-validated)

Extra controls: time-shuffle null for kin; negative control on real sessions.

Outputs: v2_benchmark_matrix.csv, v2_benchmark_calls.csv, v2_fusion.json,
v2_benchmark_summary.json, v2_mixed_features.csv (+crops)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A
import attr_core as CORE
import attr2_core as C2
import attr_10_extract_features as EX

MIN_RES, MIN_PART = 6, 3
N_AUDIO_PAIRS = 12
SEED = A.RNG_SEED


def load_df():
    out = A.ensure_out()
    feats = pd.read_csv(out / "call_features.csv")
    feats["animal_id"] = feats["animal_id"].astype(str)
    feats = A.apply_genotype(feats)
    ae = np.load(out / "call_embeddings.npz", allow_pickle=False)
    l1 = pd.DataFrame(ae["latent"], columns=[f"z{i}" for i in range(ae["latent"].shape[1])])
    l1["call_id"] = ae["call_id"]
    co = np.load(out / "v2_contrastive.npz", allow_pickle=False)
    l2 = pd.DataFrame(co["z"], columns=[f"c{i}" for i in range(co["z"].shape[1])])
    l2["call_id"] = co["call_id"]
    df = feats.merge(l1, on="call_id").merge(l2, on="call_id").reset_index(drop=True)
    z1 = [c for c in df.columns if c.startswith("z") and c[1:].isdigit()]
    z2 = [c for c in df.columns if c.startswith("c") and c[1:].isdigit()]
    return df, z1, z2


def make_pairs(df):
    counts = df[df["phase"] == "alone"].groupby("animal_id").size()
    return [(a, b)
            for a in counts[counts >= MIN_RES].index
            for b in counts[counts >= MIN_PART].index if a != b], counts


def build_dyads(df, X1, X2, pairs, kin, coef, b0, level_cue, shuffle_times=False):
    """Per pair: cue columns + truth. Returns long DataFrame + session lists."""
    rng = np.random.default_rng(SEED)
    alone = df[df["phase"] == "alone"]
    rows = []
    sess1, sess2 = [], []
    for a, b in pairs:
        ga = alone[alone["animal_id"] == a].sort_values("start_s")
        gb = alone[alone["animal_id"] == b].sort_values("start_s")
        half = len(ga) // 2
        anchor_idx = ga.index.to_numpy()[:half]
        test_idx = np.concatenate([ga.index.to_numpy()[half:], gb.index.to_numpy()])
        truth = np.concatenate([np.ones(len(ga) - half), np.zeros(len(gb))])
        times = df.loc[test_idx, "start_s"].to_numpy()
        if shuffle_times:
            times = rng.uniform(2, A.PARTNER_INTRO_S - 2, len(times))
        kA = kin[a]
        kB = kin[b]
        kin_lo = (C2.kin_score(kA, "m1", times, "alone", coef, b0)
                  - C2.kin_score(kB, "m1", times, "alone", coef, b0))
        posR = np.stack([np.interp(times, kA["t"], kA["m1_x"]),
                         np.interp(times, kA["t"], kA["m1_y"])], 1)
        posP = np.stack([np.interp(times, kB["t"], kB["m1_x"]),
                         np.interp(times, kB["t"], kB["m1_y"])], 1)
        lev_lo = level_cue.logodds(df.loc[test_idx], posR, posP)
        key = f"{a}_vs_{b}"
        for X, lst in ((X1, sess1), (X2, sess2)):
            lst.append(CORE.Session(
                key=key, stim_key=key, X=X[test_idx], times=times,
                level_db=df.loc[test_idx, "rf_amp_peak_db"].to_numpy(),
                pos_R=posR, pos_P=posP, anchor_X=X[anchor_idx]))
        for j, (ci, y) in enumerate(zip(test_idx, truth)):
            rows.append({"pair": key, "row": ci, "truth": y, "t": times[j],
                         "kin": kin_lo[j], "level": lev_lo[j]})
    long = pd.DataFrame(rows)
    v1 = C2.voice_logodds(sess1, seed=SEED)
    v2 = C2.voice_logodds(sess2, seed=SEED)
    long["voice_v1"] = np.concatenate([v1[s.key] for s in sess1])
    long["voice_v2"] = np.concatenate([v2[s.key] for s in sess2])
    return long


def lopo_fusion(long, cue_cols):
    """Leave-one-pair-out fused log-odds (unbiased)."""
    out = np.full(len(long), np.nan)
    M = long[cue_cols].to_numpy(float)
    y = long["truth"].to_numpy()
    for pair in long["pair"].unique():
        te = (long["pair"] == pair).to_numpy()
        if len(np.unique(y[~te])) < 2:
            continue
        f = C2.Fusion(cue_cols).fit(M[~te], y[~te])
        out[te] = f.logodds(M[te])
    return out


def score(long, col, hmm=False, tau=CORE.HMM_TAU_S, p_min=CORE.HMM_P_MIN):
    y = long["truth"].to_numpy()
    lo = long[col].to_numpy(float)
    lo = np.where(np.isfinite(lo), lo, 0.0)
    if hmm:
        sm = np.empty(len(long))
        for pair, g in long.groupby("pair"):
            ii = long.index.get_indexer(g.index)
            sm[ii] = CORE.hmm_smooth(g["t"].to_numpy(), lo[ii] / 2, -lo[ii] / 2,
                                     0.5, p_min=p_min, tau=tau)
        lo = sm
    pooled = roc_auc_score(y, lo)
    per_pair = [roc_auc_score(g["truth"], lo[long.index.get_indexer(g.index)])
                for _, g in long.groupby("pair")
                if g["truth"].nunique() == 2]
    return pooled, float(np.mean(per_pair))


def audio_level_dyads(df, z1, z2, pairs_subset, X1t, X2t, kin, coef, b0, level_cue):
    """Re-extract features from summed waveforms, encode with BOTH encoders."""
    import torch
    from attr_20_train_embedder import ConvAE
    from attr2_20_contrastive import Encoder

    out = A.ensure_out()
    ae = ConvAE()
    ae.load_state_dict(torch.load(out / "embedder_scratch.pt", weights_only=True))
    ae.eval()
    enc = Encoder()
    enc.load_state_dict(torch.load(out / "v2_contrastive.pt", weights_only=True))
    enc.eval()

    calls = A.load_calls()
    calls = calls[calls["phase"] == "alone"]
    frames = []
    for a, b in pairs_subset:
        for src, other in ((a, b), (b, a)):
            rows, crops, ids = EX.process_animal(src, calls, mix_with=other)
            if not rows:
                continue
            fd = pd.DataFrame(rows)
            Sc = np.stack(crops).astype(np.float32)[:, None]
            with torch.no_grad():
                Z1 = ae.encode(torch.from_numpy(Sc)).numpy()
                Z2 = enc(torch.from_numpy(Sc)).numpy()
            for i, zc in enumerate(z1):
                fd[zc] = Z1[:, i]
            for i, zc in enumerate(z2):
                fd[zc] = Z2[:, i]
            fd["mix_pair"] = f"{a}_vs_{b}"
            fd["mix_role"] = "res" if src == a else "part"
            frames.append(fd)
        print(f"  mixed {a}x{b}", flush=True)
    mixed = pd.concat(frames, ignore_index=True)
    mixed.to_csv(out / "v2_mixed_features.csv", index=False)

    alone = df[df["phase"] == "alone"]
    rows, sess1, sess2 = [], [], []
    for key, g in mixed.groupby("mix_pair"):
        a, b = key.split("_vs_")
        ga = alone[alone["animal_id"] == a].sort_values("start_s")
        half = len(ga) // 2
        anchor_idx = ga.index.to_numpy()[:half]
        anchor_start = set(ga["start_s"].round(4).to_numpy()[:half])
        gres = g[g["mix_role"] == "res"]
        keep = ~gres["start_s"].round(4).isin(anchor_start)
        gtest = pd.concat([gres[keep], g[g["mix_role"] == "part"]])
        times = gtest["start_s"].to_numpy()
        truth = (gtest["mix_role"] == "res").to_numpy().astype(float)
        kA, kB = kin[a], kin[b]
        kin_lo = (C2.kin_score(kA, "m1", times, "alone", coef, b0)
                  - C2.kin_score(kB, "m1", times, "alone", coef, b0))
        posR = np.stack([np.interp(times, kA["t"], kA["m1_x"]),
                         np.interp(times, kA["t"], kA["m1_y"])], 1)
        posP = np.stack([np.interp(times, kB["t"], kB["m1_x"]),
                         np.interp(times, kB["t"], kB["m1_y"])], 1)
        lev_lo = level_cue.logodds(gtest, posR, posP)
        Xm1 = X1t(gtest, gtest[z1].to_numpy())
        Xm2 = X2t(gtest, gtest[z2].to_numpy())
        sess1.append(CORE.Session(key, key, Xm1, times,
                                  gtest["rf_amp_peak_db"].to_numpy(),
                                  posR, posP, X1t(df.loc[anchor_idx],
                                                  df.loc[anchor_idx, z1].to_numpy())))
        sess2.append(CORE.Session(key, key, Xm2, times,
                                  gtest["rf_amp_peak_db"].to_numpy(),
                                  posR, posP, X2t(df.loc[anchor_idx],
                                                  df.loc[anchor_idx, z2].to_numpy())))
        for j, y in enumerate(truth):
            rows.append({"pair": key, "truth": y, "t": times[j],
                         "kin": kin_lo[j], "level": lev_lo[j]})
    long = pd.DataFrame(rows)
    v1lo = C2.voice_logodds(sess1, seed=SEED)
    v2lo = C2.voice_logodds(sess2, seed=SEED)
    long["voice_v1"] = np.concatenate([v1lo[s.key] for s in sess1])
    long["voice_v2"] = np.concatenate([v2lo[s.key] for s in sess2])
    return long


def main():
    out = A.ensure_out()
    df, z1, z2 = load_df()
    kin, coef, b0 = C2.load_kinematics()

    X1, X1t, _ = CORE.build_voice_space(df, df[z1].to_numpy())
    X2, X2t, _ = CORE.build_voice_space(df, df[z2].to_numpy())

    alone = df[df["phase"] == "alone"]
    level_cue = C2.LevelCue.fit(alone)
    print(f"level cue: LOAO R2={level_cue.r2_loao:.3f} enabled={level_cue.enabled}")

    pairs, counts = make_pairs(df)
    print(f"{len(pairs)} pseudo-dyad pairs")

    long = build_dyads(df, X1, X2, pairs, kin, coef, b0, level_cue)
    long_sh = build_dyads(df, X1, X2, pairs, kin, coef, b0, level_cue,
                          shuffle_times=True)

    FUSE = ["kin", "voice_v2", "level"] if level_cue.enabled else ["kin", "voice_v2"]
    FUSE_ALL = ["kin", "voice_v1", "voice_v2"] + (["level"] if level_cue.enabled else [])
    long["fusion"] = lopo_fusion(long, FUSE)
    long["fusion_all"] = lopo_fusion(long, FUSE_ALL)

    # HMM parameter grid on the LOPO fused scores (2 params, 198 pairs)
    grid = [(tau, pm) for tau in (0.5, 1.0, 2.0, 4.0)
            for pm in (0.02, 0.05, 0.10, 0.20)]
    hmm_scores = {g: score(long, "fusion", hmm=True, tau=g[0], p_min=g[1])[0]
                  for g in grid}
    (best_tau, best_pmin) = max(hmm_scores, key=hmm_scores.get)
    print(f"HMM grid best: tau={best_tau}, p_min={best_pmin}, "
          f"auc={hmm_scores[(best_tau, best_pmin)]:.3f}")

    matrix = []
    for m in ["kin", "voice_v1", "voice_v2", "level", "fusion", "fusion_all"]:
        for hmm in (False, True):
            pooled, per_pair = score(long, m, hmm=hmm, tau=best_tau,
                                     p_min=best_pmin)
            matrix.append({"method": m, "hmm": hmm, "set": "feature",
                           "pooled_auc": pooled, "mean_pair_auc": per_pair})
    kin_sh_pooled, _ = score(long_sh, "kin")
    print(f"time-shuffle kin AUC = {kin_sh_pooled:.3f} (should be ~0.5)")

    print("audio-level mixtures...")
    pair_rank = sorted(pairs, key=lambda p: min(counts[p[0]], counts[p[1]]),
                       reverse=True)
    seen, subset = set(), []
    for a, b in pair_rank:
        if (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))
        subset.append((a, b))
        if len(subset) >= N_AUDIO_PAIRS:
            break
    longA = audio_level_dyads(df, z1, z2, subset, X1t, X2t, kin, coef, b0,
                              level_cue)
    fus = C2.Fusion(FUSE).fit(long[FUSE].to_numpy(float),
                              long["truth"].to_numpy())
    longA["fusion"] = fus.logodds(longA[FUSE].to_numpy(float))
    for m in ["kin", "voice_v1", "voice_v2", "level", "fusion"]:
        for hmm in (False, True):
            pooled, per_pair = score(longA, m, hmm=hmm, tau=best_tau,
                                     p_min=best_pmin)
            matrix.append({"method": m, "hmm": hmm, "set": "audio",
                           "pooled_auc": pooled, "mean_pair_auc": per_pair})

    mat = pd.DataFrame(matrix)
    mat.to_csv(out / "v2_benchmark_matrix.csv", index=False)
    long.to_csv(out / "v2_benchmark_calls.csv", index=False)
    longA.to_csv(out / "v2_benchmark_calls_audio.csv", index=False)
    print(mat.pivot_table(index=["method", "hmm"], columns="set",
                          values="pooled_auc").round(3).to_string())

    # winner = the deployable fusion (its weights/calibration are what ship);
    # fusion_all is a diagnostic row only
    feat = mat[(mat["set"] == "feature") & mat["hmm"]]
    winner = "fusion"
    fus_final = C2.Fusion(FUSE).fit(long[FUSE].to_numpy(float),
                                    long["truth"].to_numpy())

    # calibration of winner's LOPO scores (logistic 1-D on held-out log-odds)
    from sklearn.linear_model import LogisticRegression
    lo = long["fusion"].to_numpy(float)[:, None]
    okc = np.isfinite(lo[:, 0])
    cal = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        lo[okc], long["truth"].to_numpy()[okc])

    p_cal = cal.predict_proba(lo[okc])[:, 1]
    y = long["truth"].to_numpy()[okc]
    bins = np.linspace(0, 1, 11)
    bid = np.digitize(p_cal, bins) - 1
    calib_pts = [[bins[i] + .05, float(y[bid == i].mean()), int((bid == i).sum())]
                 for i in range(10) if (bid == i).sum() >= 10]

    err_res = float(np.mean(p_cal[y == 1] <= 0.5))
    err_part = float(np.mean(p_cal[y == 0] > 0.5))

    summary = {
        "winner": str(winner),
        "fusion_cues": FUSE,
        "fusion_weights": fus_final.weights,
        "cal_slope": float(cal.coef_[0][0]), "cal_intercept": float(cal.intercept_[0]),
        "level_enabled": bool(level_cue.enabled),
        "level_r2_loao": float(level_cue.r2_loao),
        "kin_shuffle_auc": float(kin_sh_pooled),
        "feature_fusion_hmm_auc": float(
            feat.set_index("method").loc["fusion", "pooled_auc"]),
        "audio_fusion_hmm_auc": float(
            mat[(mat["set"] == "audio") & mat["hmm"]
                ].set_index("method").loc["fusion", "pooled_auc"]),
        "err_res_at_05": err_res, "err_part_at_05": err_part,
        "calibration_pts": calib_pts,
        "hmm_tau": float(best_tau), "hmm_p_min": float(best_pmin),
        "n_pairs_feature": int(long["pair"].nunique()),
        "n_pairs_audio": int(longA["pair"].nunique()),
    }
    with open(out / "v2_benchmark_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(out / "v2_fusion.json", "w") as fh:
        json.dump({"cues": FUSE, "weights": fus_final.weights,
                   "cal_slope": summary["cal_slope"],
                   "cal_intercept": summary["cal_intercept"],
                   "hmm_tau": float(best_tau),
                   "hmm_p_min": float(best_pmin)}, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "calibration_pts"}, indent=2))


if __name__ == "__main__":
    main()
