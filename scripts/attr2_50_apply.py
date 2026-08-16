"""V2 stage 5: apply the benchmark-selected fusion to the real sessions and
validate it on real data (no stim IDs, no table info).

Real-data validations:
1. DESIGN PSEUDO-TRUTH: v1's design bound (independent of acoustics) says
   >=92% of WT-dyad interaction calls are the resident's. The v2 model never
   sees that bound - agreement is evidence it works outside the benchmark.
2. TIME-SHUFFLE: recompute the kinematic cue with permuted call times; the
   fused attribution should lose its kinematic information.
3. NEGATIVE CONTROL: half-anchor refit; held-out alone calls (known resident)
   scored by the fused model (kin neutral there since no partner exists).
4. TWO-VOICE OVERLAP: calls showing two simultaneous non-harmonic bands are
   direct evidence of partner vocalization; their per-session counts should
   track partner-attributed fractions.

Outputs: v2_call_attribution.csv, v2_validation.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A
import attr_core as CORE
import attr2_core as C2
from attr2_40_benchmarks import load_df

SEED = A.RNG_SEED
ABSTAIN_LO, ABSTAIN_HI = 0.35, 0.65


def real_sessions(df, X, anchor_frac=1.0):
    sessions, index = [], {}
    rng = np.random.default_rng(SEED)
    for animal, g in df.groupby("animal_id"):
        part = g[g["phase"] == "partner"]
        anch = g[g["phase"] == "alone"].sort_values("start_s")
        n_anchor = max(1, int(len(anch) * anchor_frac)) if len(anch) else 0
        a_idx = anch.index.to_numpy()[:n_anchor]
        sessions.append(CORE.Session(
            key=animal, stim_key=animal, X=X[part.index.to_numpy()],
            times=part["start_s"].to_numpy(),
            level_db=part["rf_amp_peak_db"].to_numpy(),
            pos_R=part[["m1_x", "m1_y"]].to_numpy(),
            pos_P=part[["m2_x", "m2_y"]].to_numpy(),
            anchor_X=X[a_idx]))
        index[animal] = part.index.to_numpy()
    return sessions, index


def compute_cues(df, X2, X1, kin, coef, b0, level_cue, fuse_cues,
                 shuffle_times=False):
    """Cue table for all partner-phase calls of all real sessions."""
    rng = np.random.default_rng(SEED + 1)
    sess2, idx = real_sessions(df, X2)
    v2 = C2.voice_logodds(sess2, seed=SEED)
    need_v1 = "voice_v1" in fuse_cues
    if need_v1:
        sess1, _ = real_sessions(df, X1)
        v1 = C2.voice_logodds(sess1, seed=SEED)
    rows = []
    for s in sess2:
        animal = s.key
        g = df.loc[idx[animal]]
        times = g["start_s"].to_numpy()
        tk = times if not shuffle_times else rng.uniform(
            A.PARTNER_INTRO_S + 2, A.SESSION_END_S - 2, len(times))
        k = kin[animal]
        kin_lo = (C2.kin_score(k, "m1", tk, "partner", coef, b0)
                  - C2.kin_score(k, "m2", tk, "partner", coef, b0))
        lev_lo = level_cue.logodds(g, g[["m1_x", "m1_y"]].to_numpy(),
                                   g[["m2_x", "m2_y"]].to_numpy())
        d = pd.DataFrame({
            "call_id": g["call_id"].to_numpy(), "animal_id": animal,
            "genotype": g["genotype"].to_numpy(), "t": times,
            "kin": kin_lo, "level": lev_lo, "voice_v2": v2[animal],
        })
        if need_v1:
            d["voice_v1"] = v1[animal]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main():
    out = A.ensure_out()
    df, z1, z2 = load_df()
    kin, coef, b0 = C2.load_kinematics()
    with open(out / "v2_fusion.json") as fh:
        fus_info = json.load(fh)
    FUSE = fus_info["cues"]
    w = fus_info["weights"]
    cal_a, cal_b = fus_info["cal_slope"], fus_info["cal_intercept"]
    hmm_tau = fus_info.get("hmm_tau", CORE.HMM_TAU_S)
    hmm_pmin = fus_info.get("hmm_p_min", CORE.HMM_P_MIN)

    X1, _, _ = CORE.build_voice_space(df, df[z1].to_numpy())
    X2, _, _ = CORE.build_voice_space(df, df[z2].to_numpy())
    level_cue = C2.LevelCue.fit(df[df["phase"] == "alone"])

    def fuse(tab):
        lo = np.full(len(tab), w["intercept"], float)
        for c in FUSE:
            v = tab[c].to_numpy(float)
            lo += w[c] * np.where(np.isfinite(v), v, 0.0)
        return lo

    def posteriors(tab):
        lo = fuse(tab)
        p_cal = 1 / (1 + np.exp(-(cal_a * lo + cal_b)))
        p_hmm = np.empty(len(tab))
        for animal, g in tab.groupby("animal_id"):
            ii = tab.index.get_indexer(g.index)
            lo_cal = cal_a * lo[ii] + cal_b
            p_hmm[ii] = CORE.hmm_smooth(g["t"].to_numpy(), lo_cal / 2,
                                        -lo_cal / 2, 0.5,
                                        p_min=hmm_pmin, tau=hmm_tau)
        return lo, p_cal, p_hmm

    tab = compute_cues(df, X2, X1, kin, coef, b0, level_cue, FUSE)
    lo, p_cal, p_hmm = posteriors(tab)
    tab["fused_logodds"] = lo
    tab["p_res_cal"] = p_cal
    tab["p_res_hmm"] = p_hmm
    tab["abstain"] = (p_hmm > ABSTAIN_LO) & (p_hmm < ABSTAIN_HI)

    # overlap scores from stored crops
    specs = np.load(out / "call_specs.npz", allow_pickle=False)
    ov = C2.overlap_scores(specs["specs"].astype(np.float32))
    ovmap = dict(zip(specs["call_id"], ov))
    tab["overlap_score"] = tab["call_id"].map(ovmap)
    tab.to_csv(out / "v2_call_attribution.csv", index=False)

    # ---- validations ------------------------------------------------------
    val = {}
    wt = tab[tab["genotype"] == "WT"]
    per_wt = wt.groupby("animal_id")["p_res_hmm"].mean()
    val["wt_resident_frac_mean"] = float(per_wt.mean())
    val["wt_resident_frac_per_session"] = per_wt.round(3).to_dict()
    val["design_bound_wt_resident_frac"] = 0.916  # from v1 design analysis
    het = tab[tab["genotype"] == "HET"]
    val["het_resident_frac_mean"] = float(
        het.groupby("animal_id")["p_res_hmm"].mean().mean())
    val["abstain_frac"] = float(tab["abstain"].mean())

    tab_sh = compute_cues(df, X2, X1, kin, coef, b0, level_cue, FUSE,
                          shuffle_times=True)
    _, _, p_sh = posteriors(tab_sh)
    val["shuffle_wt_resident_frac"] = float(
        tab_sh.assign(p=p_sh)[tab_sh["genotype"] == "WT"]
        .groupby("animal_id")["p"].mean().mean())

    # negative control: half anchors, held-out alone calls, voice+level only
    sess_half, _ = real_sessions(df, X2, anchor_frac=0.5)
    res_half = CORE.fit_attribution(sess_half, imodel=None, seed=SEED)
    g_ = res_half["_globals"]
    nc_pass, nc_n = 0, 0
    for s in sess_half:
        anch = df[(df["animal_id"] == s.key) & (df["phase"] == "alone")
                  ].sort_values("start_s")
        if len(anch) < 4:
            continue
        held = anch.index.to_numpy()[len(anch) // 2:]
        Xh = X2[held]
        llR = CORE._gauss_logpdf_diag(Xh, res_half[s.key]["muR"], g_["varR"])
        llP = CORE._gauss_logpdf_diag(Xh, res_half[s.key]["muP"], g_["varP"])
        lo_nc = w["intercept"] + w.get("voice_v2", 0.0) * (llR - llP)
        nc_pass += int((lo_nc > 0).sum())
        nc_n += len(held)
    val["negative_control_frac"] = nc_pass / max(nc_n, 1)
    val["negative_control_n"] = nc_n

    # overlap consistency
    ses_ov = tab.groupby("animal_id").agg(
        part_frac=("p_res_hmm", lambda p: float((1 - p).mean())),
        ov_rate=("overlap_score", lambda s: float((s > 0.2).mean())),
        n=("call_id", "size"))
    big = ses_ov[ses_ov["n"] >= 10]
    if len(big) >= 4:
        val["overlap_vs_partnerfrac_spearman"] = float(
            big["part_frac"].corr(big["ov_rate"], method="spearman"))
    val["overlap_call_frac_partner_phase"] = float((tab["overlap_score"] > 0.2).mean())

    with open(out / "v2_validation.json", "w") as fh:
        json.dump(val, fh, indent=2)
    print(json.dumps(val, indent=2))


if __name__ == "__main__":
    main()
