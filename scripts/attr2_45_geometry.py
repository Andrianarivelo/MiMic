"""V2 final cue: social-geometry attribution (GEO), trained on real WT
sessions with the design prior, validated leave-one-session-out.

Why: pseudo-dyad benchmarks showed movement identifies callers when the two
animals move independently, but real interacting mice have correlated
kinematics - the per-animal cue dies (real-data shuffle control caught it).
What survives in real interactions is GEOMETRY ASYMMETRY: calls are emitted
at close range while the caller PURSUES and the other animal RETREATS
(diagnostic on this dataset: resident approach +83 px/s at WT call times vs
+34 baseline; partner -90 vs -36; the same pattern holds in HET sessions).

Model: caller-centric emission logistic f([app_self, app_other, speed_self,
speed_other, dist]) trained on WT partner-phase calls (resident is the true
caller for >=92% of them - design bound from genotype labels only, stim table
never used). Attribution score per call = f(resident-centric) -
f(partner-centric); antisymmetric by construction, calibrated with the
mirror trick, HMM-smoothed with benchmark-tuned parameters.

Validation (all on real data):
  LOSO   leave-one-WT-session-out: held-out sessions must recover a high
         resident fraction from geometry alone
  SHUF   time-shuffled call times must destroy the attribution signal
  DIR    learned weights must match the literature (pursuit+, retreat-)

Outputs: v2_geo_attribution.csv, v2_geo_validation.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A
import attr_core as CORE
import attr2_core as C2

SEED = A.RNG_SEED
PI_WT = 0.916          # design bound: resident share of WT dyad calls
HMM_TAU, HMM_PMIN = 0.5, 0.2
GEO_FEATS = ["app_self", "app_other", "speed_self", "speed_other", "dist_z"]


def session_geometry(k: pd.DataFrame, times: np.ndarray) -> dict:
    """Raw geometry at given times: positions, speeds, approach velocities."""
    t = k["t"].to_numpy()

    def gi(col, tt):
        v = k[col].to_numpy()
        ok = np.isfinite(v)
        return (np.interp(tt, t[ok], v[ok]) if ok.sum() > 10
                else np.full(len(tt), np.nan))

    x1, y1 = gi("m1_x", times), gi("m1_y", times)
    x2, y2 = gi("m2_x", times), gi("m2_y", times)
    dist = np.hypot(x1 - x2, y1 - y2)
    dt = 0.15

    def vel(prefix):
        vx = (gi(f"{prefix}_x", times + dt) - gi(f"{prefix}_x", times - dt)) / (2 * dt)
        vy = (gi(f"{prefix}_y", times + dt) - gi(f"{prefix}_y", times - dt)) / (2 * dt)
        return vx, vy

    v1x, v1y = vel("m1")
    v2x, v2y = vel("m2")
    ux, uy = (x2 - x1) / (dist + 1e-9), (y2 - y1) / (dist + 1e-9)
    return dict(
        dist=dist,
        s1=gi("m1_speed", times), s2=gi("m2_speed", times),
        app1=v1x * ux + v1y * uy,       # m1 toward m2
        app2=-(v2x * ux + v2y * uy),    # m2 toward m1
    )


def caller_centric(geo: dict, zs: dict, caller: str) -> np.ndarray:
    """Feature matrix from one candidate caller's perspective, z-scored with
    SESSION-SHARED (symmetric) statistics so the mirror flip is exact."""
    if caller == "m1":
        app_s, app_o, sp_s, sp_o = geo["app1"], geo["app2"], geo["s1"], geo["s2"]
    else:
        app_s, app_o, sp_s, sp_o = geo["app2"], geo["app1"], geo["s2"], geo["s1"]
    cols = [
        (app_s - zs["app_mu"]) / zs["app_sd"],
        (app_o - zs["app_mu"]) / zs["app_sd"],
        (sp_s - zs["sp_mu"]) / zs["sp_sd"],
        (sp_o - zs["sp_mu"]) / zs["sp_sd"],
        (geo["dist"] - zs["d_mu"]) / zs["d_sd"],
    ]
    return np.stack(cols, axis=1)


def session_zstats(k: pd.DataFrame) -> dict:
    """Symmetric z-stats over the partner phase (both animals pooled)."""
    t = k["t"].to_numpy()
    m = t >= A.PARTNER_INTRO_S
    grid = t[m][::3]
    geo = session_geometry(k, grid)
    app = np.concatenate([geo["app1"], geo["app2"]])
    sp = np.concatenate([geo["s1"], geo["s2"]])
    return dict(app_mu=np.nanmean(app), app_sd=np.nanstd(app) + 1e-9,
                sp_mu=np.nanmean(sp), sp_sd=np.nanstd(sp) + 1e-9,
                d_mu=np.nanmean(geo["dist"]), d_sd=np.nanstd(geo["dist"]) + 1e-9)


def build_training(kin, calls, animals, rng):
    """WT-session training rows: (features, y, weight, session)."""
    X, y, w, sess = [], [], [], []
    for a in animals:
        k = kin[a]
        zs = session_zstats(k)
        g = calls[(calls["animal_id"] == a) & (calls["phase"] == "partner")]
        if len(g) < 3:
            continue
        ct = ((g["start_s"] + g["end_s"]) / 2).to_numpy()
        geo_c = session_geometry(k, ct)
        ctrl = rng.uniform(A.PARTNER_INTRO_S + 2, A.SESSION_END_S - 2, 3 * len(ct))
        geo_n = session_geometry(k, ctrl)
        blocks = [
            (caller_centric(geo_c, zs, "m1"), 1.0, PI_WT),        # calls, res view
            (caller_centric(geo_c, zs, "m2"), 0.0, PI_WT),        # calls, part view
            (caller_centric(geo_n, zs, "m1"), 0.0, 0.5),          # controls
            (caller_centric(geo_n, zs, "m2"), 0.0, 0.5),
        ]
        for M, lab, wt in blocks:
            X.append(M)
            y.append(np.full(len(M), lab))
            w.append(np.full(len(M), wt))
            sess.append(np.repeat(a, len(M)))
    X = np.vstack(X)
    y = np.concatenate(y)
    w = np.concatenate(w)
    sess = np.concatenate(sess)
    ok = np.isfinite(X).all(axis=1)
    return X[ok], y[ok], w[ok], sess[ok]


def geo_logodds(kin, animal, times, clf) -> np.ndarray:
    k = kin[animal]
    zs = session_zstats(k)
    geo = session_geometry(k, times)
    s1 = clf.decision_function(np.nan_to_num(caller_centric(geo, zs, "m1")))
    s2 = clf.decision_function(np.nan_to_num(caller_centric(geo, zs, "m2")))
    lo = s1 - s2
    bad = ~np.isfinite(caller_centric(geo, zs, "m1")).all(axis=1)
    lo[bad] = 0.0
    return lo


def fit_clf(X, y, w):
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        X, y, sample_weight=w)


def mirror_calibration(scores: np.ndarray, pi: float = PI_WT):
    """Fit sigmoid slope on {(s,1) w=pi, (-s,0) w=pi, (s,0) w=1-pi, (-s,1) w=1-pi}."""
    s = scores[np.isfinite(scores)]
    Xc = np.concatenate([s, -s, s, -s])[:, None]
    yc = np.concatenate([np.ones_like(s), np.zeros_like(s),
                         np.zeros_like(s), np.ones_like(s)])
    wc = np.concatenate([np.full_like(s, pi), np.full_like(s, pi),
                         np.full_like(s, 1 - pi), np.full_like(s, 1 - pi)])
    cal = LogisticRegression(max_iter=1000).fit(Xc, yc, sample_weight=wc)
    return float(cal.coef_[0][0]), float(cal.intercept_[0])


def pi_mle(cal_lo: np.ndarray):
    """Session resident-prevalence MLE from calibrated per-call log-LRs,
    with a profile-likelihood 95% CI. Mean-of-posteriors underestimates
    prevalence when per-call evidence is weak; the MLE does not."""
    s = cal_lo[np.isfinite(cal_lo)]
    if not len(s):
        return np.nan, (np.nan, np.nan)
    lr = np.exp(np.clip(s, -30, 30))
    grid = np.linspace(0.001, 0.999, 500)
    ll = np.array([np.sum(np.log(p * lr + (1 - p))) for p in grid])
    i = int(np.argmax(ll))
    inside = grid[ll >= ll[i] - 1.92]
    return float(grid[i]), (float(inside.min()), float(inside.max()))


def train_scores(kin, calls, animals, clf):
    out = []
    for a in animals:
        g = calls[(calls["animal_id"] == a) & (calls["phase"] == "partner")]
        if not len(g):
            continue
        ct = ((g["start_s"] + g["end_s"]) / 2).to_numpy()
        out.append(geo_logodds(kin, a, ct, clf))
    return np.concatenate(out)


def main():
    out = A.ensure_out()
    rng = np.random.default_rng(SEED)
    kin, _, _ = C2.load_kinematics()
    calls = A.load_calls()
    wt_animals = [a for a, g in A.GENOTYPE_MAP.items() if g == "WT"]

    X, y, w, sess = build_training(kin, calls, wt_animals, rng)
    print(f"training rows: {len(X)} from {len(np.unique(sess))} WT sessions")

    # ---- LOSO validation (fold-internal calibration, prevalence MLE) ------
    loso_pi, shuf_pi, loso_mean_post = {}, {}, {}
    for hold in np.unique(sess):
        tr = sess != hold
        clf = fit_clf(X[tr], y[tr], w[tr])
        cal_a_f, _ = mirror_calibration(
            train_scores(kin, calls, [a for a in wt_animals if a != hold], clf))
        g = calls[(calls["animal_id"] == hold) & (calls["phase"] == "partner")]
        ct = ((g["start_s"] + g["end_s"]) / 2).to_numpy()
        lo = geo_logodds(kin, hold, ct, clf)
        loso_pi[hold] = pi_mle(cal_a_f * lo)[0]
        p = CORE.hmm_smooth(ct, lo / 2, -lo / 2, 0.5, p_min=HMM_PMIN, tau=HMM_TAU)
        loso_mean_post[hold] = float(np.mean(p))
        sh = np.sort(rng.uniform(A.PARTNER_INTRO_S + 2, A.SESSION_END_S - 2,
                                 len(ct)))
        shuf_pi[hold] = pi_mle(cal_a_f * geo_logodds(kin, hold, sh, clf))[0]
    loso_mean = float(np.nanmean(list(loso_pi.values())))
    shuf_mean = float(np.nanmean(list(shuf_pi.values())))
    print(f"LOSO WT resident prevalence MLE: {loso_mean:.3f} "
          f"(design bound {PI_WT}); shuffled: {shuf_mean:.3f}; "
          f"mean-posterior (biased): "
          f"{np.mean(list(loso_mean_post.values())):.3f}")

    # ---- final model + calibration ---------------------------------------
    clf = fit_clf(X, y, w)
    coefs = dict(zip(GEO_FEATS, clf.coef_[0].round(3).tolist()))
    print("GEO weights:", coefs)

    wt_scores = []
    for a in wt_animals:
        g = calls[(calls["animal_id"] == a) & (calls["phase"] == "partner")]
        if not len(g):
            continue
        ct = ((g["start_s"] + g["end_s"]) / 2).to_numpy()
        wt_scores.append(geo_logodds(kin, a, ct, clf))
    cal_a, cal_b = mirror_calibration(np.concatenate(wt_scores))

    # sensitivity at 0.5, corrected for the 8.4% label noise in WT "truth"
    s_all = np.concatenate(wt_scores)
    obs = float(np.mean(s_all > 0))
    sens = float(np.clip((obs - (1 - PI_WT) * 0.5) / PI_WT, 0, 1))

    # ---- apply to all sessions -------------------------------------------
    # per-session: prevalence MLE + CI; per-call posterior under that prior
    rows, ses_rows = [], []
    for animal, gt in A.GENOTYPE_MAP.items():
        g = calls[(calls["animal_id"] == animal) & (calls["phase"] == "partner")]
        if not len(g):
            ses_rows.append({"animal_id": animal, "genotype": gt, "n": 0,
                             "pi_mle": np.nan, "pi_lo": np.nan,
                             "pi_hi": np.nan, "pi_shuffle": np.nan})
            continue
        ct = ((g["start_s"] + g["end_s"]) / 2).to_numpy()
        lo = geo_logodds(kin, animal, ct, clf)
        lo_cal = cal_a * lo
        pi_hat, (pi_lo, pi_hi) = pi_mle(lo_cal)
        # session-specific shuffle null: what chronic pursuit asymmetry alone
        # (random times) would imply, separated from call-locked evidence
        sh = np.sort(rng.uniform(A.PARTNER_INTRO_S + 2, A.SESSION_END_S - 2,
                                 max(len(ct) * 3, 30)))
        pi_sh, _ = pi_mle(cal_a * geo_logodds(kin, animal, sh, clf))
        ses_rows.append({"animal_id": animal, "genotype": gt, "n": len(g),
                         "pi_mle": pi_hat, "pi_lo": pi_lo, "pi_hi": pi_hi,
                         "pi_shuffle": pi_sh})
        prior = np.log(np.clip(pi_hat, 1e-3, 1 - 1e-3) /
                       (1 - np.clip(pi_hat, 1e-3, 1 - 1e-3)))
        lo_post = lo_cal + prior
        p_cal = 1 / (1 + np.exp(-lo_post))
        p_hmm = CORE.hmm_smooth(ct, lo_post / 2, -lo_post / 2, 0.5,
                                p_min=HMM_PMIN, tau=HMM_TAU)
        for cid, t_, l_, pc, ph in zip(g["call_id"], ct, lo, p_cal, p_hmm):
            rows.append({"call_id": cid, "animal_id": animal, "genotype": gt,
                         "t": t_, "geo_logodds": l_, "p_res_cal": pc,
                         "p_res_hmm": ph,
                         "abstain": bool(0.35 < ph < 0.65)})
    tab = pd.DataFrame(rows)
    pd.DataFrame(ses_rows).to_csv(out / "v2_geo_sessions.csv", index=False)
    # keep overlap scores alongside
    prev = pd.read_csv(out / "v2_call_attribution.csv")[["call_id", "overlap_score"]]
    tab = tab.merge(prev, on="call_id", how="left")
    tab.to_csv(out / "v2_geo_attribution.csv", index=False)

    ses_df = pd.DataFrame(ses_rows)
    het_pi = ses_df[(ses_df["genotype"] == "HET") & (ses_df["n"] >= 3)]
    val = {
        "loso_wt_resident_pi_mle_mean": loso_mean,
        "loso_wt_resident_pi_per_session": {k: round(v, 3) for k, v in loso_pi.items()},
        "loso_wt_mean_posterior_biased": float(
            np.mean(list(loso_mean_post.values()))),
        "shuffle_wt_resident_pi_mean": shuf_mean,
        "het_pi_mle_mean": float(het_pi["pi_mle"].mean()) if len(het_pi) else None,
        "design_bound": PI_WT,
        "geo_weights": coefs,
        "cal_slope": cal_a, "cal_intercept": cal_b,
        "wt_sensitivity_at_05": sens,
        "err_at_05": float(1 - sens),
        "het_resident_frac_mean": float(
            tab[tab["genotype"] == "HET"].groupby("animal_id")["p_res_hmm"]
            .mean().mean()),
        "abstain_frac": float(tab["abstain"].mean()),
        "overlap_vs_partnerfrac_spearman": None,
    }
    ses_ov = tab.groupby("animal_id").agg(
        part_frac=("p_res_hmm", lambda p: float((1 - p).mean())),
        ov=("overlap_score", lambda s: float((s > 0.2).mean())),
        n=("call_id", "size"))
    big = ses_ov[ses_ov["n"] >= 10]
    if len(big) >= 4:
        val["overlap_vs_partnerfrac_spearman"] = float(
            big["part_frac"].corr(big["ov"], method="spearman"))
    with open(out / "v2_geo_validation.json", "w") as fh:
        json.dump(val, fh, indent=2)
    print(json.dumps(val, indent=2))


if __name__ == "__main__":
    main()
