"""Is the post-call rise in partner approach velocity real, or regression to
the mean?

The concern (raised on inspection of v4_fig3): calls concentrate during
anogenital sniffing, a state in which the resident pursues and the partner is
typically moving AWAY. A moment sampled at extreme negative approach velocity
must drift upward afterwards whatever the call does. The original matching
used behaviour state x speed x distance but NOT the partner's approach
velocity itself, so it cannot rule this out.

This script:
  1. BALANCE  standardised mean differences between calls and their matched
     controls for every pre-call covariate.
  2. TRAJECTORY  absolute (not differenced) peri-event traces for calls and
     matched controls. If the two curves sit on top of each other before the
     call and separate after it, the effect is real; if the call curve starts
     lower and merely converges, it is regression to the mean.
  3. CONTEXT  approach velocity within each behavioural state, to check
     whether the partner really is retreating during anogenital sniffing.
  4. RE-TEST  matching additionally on pre-call approach velocity and partner
     speed, plus variants using only isolated calls and only bout-onset calls.

Outputs: beh_diag_balance.csv, beh_diag_traj.csv, beh_diag_context.csv,
         beh_diag_retest.csv, beh_diag_summary.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A
from attr4_00_build_behavior import BEH, BIN_S
from attr4_10_analysis import load_bins, bh_fdr, PARTNER_T0

SEED = A.RNG_SEED
PRE = (-1.0, -0.2)
POST = (0.2, 1.0)
TRAJ = (-3.0, 5.0)
CONF = 0.65
METRICS = ["app_m2", "app_m1", "speed_m1", "speed_m2", "dist"]


def qbin(v, q):
    v = np.where(np.isfinite(v), v, np.nanmedian(v))
    return np.digitize(v, np.nanquantile(v, q))


def build_pairs(df, match_extra=False, subset="all", rng=None):
    """Return (call_idx, ctrl_idx, animal) triples under a matching scheme."""
    pre0, pre1 = int(PRE[0] / BIN_S), int(PRE[1] / BIN_S)
    t0, t1 = int(TRAJ[0] / BIN_S), int(TRAJ[1] / BIN_S)
    guard = int(1.0 / BIN_S)
    out = []
    store = {}
    for animal, g in df[df["t"] >= PARTNER_T0].groupby("animal_id"):
        g = g.sort_values("bin").reset_index(drop=True)
        n = len(g)
        counts = g["n_calls"].to_numpy()
        pres = g["p_res"].to_numpy()
        sp = g["speed_m1"].to_numpy()
        di = g["dist"].to_numpy()
        top = g["behavior_top"].fillna("none").astype(str).to_numpy()
        # pre-call value of the covariates we may want to match on
        appm2 = g["app_m2"].to_numpy()
        spm2 = g["speed_m2"].to_numpy()
        pre_app = np.array([np.nanmean(appm2[max(i + pre0, 0):i + pre1])
                            if i + pre0 >= 0 else np.nan for i in range(n)])
        pre_spm2 = np.array([np.nanmean(spm2[max(i + pre0, 0):i + pre1])
                             if i + pre0 >= 0 else np.nan for i in range(n)])
        key = np.array([f"{a}|{b}{c}" for a, b, c in
                        zip(top, qbin(sp, [.2, .4, .6, .8]), qbin(di, [1/3, 2/3]))])
        if match_extra:
            key = np.array([f"{k}|{d}{e}" for k, d, e in
                            zip(key, qbin(pre_app, [.2, .4, .6, .8]),
                                qbin(pre_spm2, [1/3, 2/3]))])
        has = counts > 0
        near = np.convolve(has.astype(int), np.ones(2 * guard + 1),
                           mode="same") > 0
        pool = {}
        for i in np.where(~near)[0]:
            if -t0 <= i < n - t1:
                pool.setdefault(key[i], []).append(i)
        idx = np.where(has)[0]
        idx = idx[(idx >= -t0) & (idx < n - t1)]
        idx = idx[pres[idx] >= CONF]
        if subset == "isolated":
            iso = np.convolve(has.astype(int), np.ones(2 * guard + 1),
                              mode="same")
            idx = idx[iso[idx] <= 1]
        elif subset == "bout_onset":
            prev = np.convolve(has.astype(int),
                               np.r_[np.ones(guard), 0, np.zeros(guard)],
                               mode="same")
            idx = idx[prev[idx] == 0]
        for i in idx:
            cands = pool.get(key[i], [])
            if len(cands) < 2:
                continue
            ctrl = rng.choice(cands, size=min(3, len(cands)), replace=False)
            out.append((animal, int(i), [int(c) for c in ctrl]))
        store[animal] = g
    return out, store


def window_mean(arr, i, w):
    a, b = int(w[0] / BIN_S), int(w[1] / BIN_S)
    lo, hi = max(i + a, 0), min(i + b, len(arr))
    return np.nanmean(arr[lo:hi]) if hi > lo else np.nan


def main():
    out = A.ensure_out()
    rng = np.random.default_rng(SEED)
    df = load_bins()

    # ---------------------------------------------------- 1. balance + traj
    pairs, store = build_pairs(df, match_extra=False, rng=rng)
    print(f"{len(pairs)} resident-attributed calls with matched controls")

    lags = np.arange(int(TRAJ[0] / BIN_S), int(TRAJ[1] / BIN_S) + 1)
    traj_call = {m: [] for m in METRICS}
    traj_ctrl = {m: [] for m in METRICS}
    bal_rows = []
    for animal, i, ctrl in pairs:
        g = store[animal]
        for m in METRICS:
            v = g[m].to_numpy()
            traj_call[m].append(v[i + lags])
            traj_ctrl[m].append(np.nanmean([v[c + lags] for c in ctrl], axis=0))
            bal_rows.append({"animal_id": animal, "metric": m,
                             "pre_call": window_mean(v, i, PRE),
                             "pre_ctrl": np.nanmean([window_mean(v, c, PRE)
                                                     for c in ctrl])})
    bal = pd.DataFrame(bal_rows)
    bal_stats = []
    for m, g in bal.groupby("metric"):
        d = g["pre_call"] - g["pre_ctrl"]
        sd = np.sqrt((g["pre_call"].var() + g["pre_ctrl"].var()) / 2)
        per = g.groupby("animal_id").apply(
            lambda x: (x["pre_call"] - x["pre_ctrl"]).mean(), include_groups=False)
        bal_stats.append({
            "metric": m, "pre_call_mean": g["pre_call"].mean(),
            "pre_ctrl_mean": g["pre_ctrl"].mean(),
            "diff": d.mean(), "smd": d.mean() / (sd + 1e-9),
            "p_wilcoxon": stats.wilcoxon(per)[1] if len(per) > 5 else np.nan,
        })
    bal_df = pd.DataFrame(bal_stats)
    bal_df.to_csv(out / "beh_diag_balance.csv", index=False)
    print("\nBALANCE (pre-call window, call vs matched control):")
    print(bal_df.round(3).to_string(index=False))

    rows = []
    for m in METRICS:
        C = np.array(traj_call[m])
        K = np.array(traj_ctrl[m])
        for k, L in enumerate(lags):
            rows.append({"metric": m, "lag_s": L * BIN_S,
                         "call_mean": np.nanmean(C[:, k]),
                         "call_sem": np.nanstd(C[:, k]) / np.sqrt(len(C)),
                         "ctrl_mean": np.nanmean(K[:, k]),
                         "ctrl_sem": np.nanstd(K[:, k]) / np.sqrt(len(K))})
    pd.DataFrame(rows).to_csv(out / "beh_diag_traj.csv", index=False)

    # ------------------------------------------------------- 3. context
    part = df[df["t"] >= PARTNER_T0]
    ctx = []
    for b in BEH + ["ACTIVEsocial"]:
        col = f"m1_{b}"
        if col not in part.columns:
            if b == "ACTIVEsocial":
                m = np.zeros(len(part), bool)
                for bb in ("nose2anogenital", "nose2body", "following",
                           "chasing", "approach"):
                    m |= part[f"m1_{bb}"].to_numpy() > 0
            else:
                continue
        else:
            m = part[col].to_numpy() > 0
        if m.sum() < 50:
            continue
        ctx.append({"behavior": b, "n_bins": int(m.sum()),
                    "app_m2_in": float(np.nanmean(part["app_m2"].to_numpy()[m])),
                    "app_m2_out": float(np.nanmean(part["app_m2"].to_numpy()[~m])),
                    "app_m1_in": float(np.nanmean(part["app_m1"].to_numpy()[m])),
                    "dist_in": float(np.nanmean(part["dist"].to_numpy()[m])),
                    "calls": int(part.loc[m, "n_calls"].sum())})
    ctx_df = pd.DataFrame(ctx).sort_values("app_m2_in")
    ctx_df.to_csv(out / "beh_diag_context.csv", index=False)
    print("\nCONTEXT: partner approach velocity inside each resident behaviour")
    print(ctx_df.round(2).to_string(index=False))

    # -------------------------------------------------------- 4. re-test
    retest = []
    for extra in (False, True):
        for subset in ("all", "isolated", "bout_onset"):
            pr, st = build_pairs(df, match_extra=extra, subset=subset,
                                 rng=np.random.default_rng(SEED))
            if len(pr) < 50:
                retest.append({"matching": "state+speed+dist" + ("+app+spd2" if extra else ""),
                               "subset": subset, "n_calls": len(pr)})
                continue
            recs = []
            for animal, i, ctrl in pr:
                g = st[animal]
                for m in METRICS:
                    v = g[m].to_numpy()
                    dc = window_mean(v, i, POST) - window_mean(v, i, PRE)
                    dk = np.nanmean([window_mean(v, c, POST) - window_mean(v, c, PRE)
                                     for c in ctrl])
                    if np.isfinite(dc) and np.isfinite(dk):
                        recs.append({"animal_id": animal, "metric": m,
                                     "diff": dc - dk})
            R = pd.DataFrame(recs)
            for m, g2 in R.groupby("metric"):
                animals = g2["animal_id"].unique()
                by = {a: g2.loc[g2["animal_id"] == a, "diff"].to_numpy()
                      for a in animals}
                boot = np.array([np.concatenate(
                    [by[a] for a in np.random.default_rng(SEED + b)
                     .choice(animals, len(animals), replace=True)]).mean()
                    for b in range(1000)])
                lo, hi = np.percentile(boot, [2.5, 97.5])
                retest.append({
                    "matching": "state+speed+dist" + ("+app+spd2" if extra else ""),
                    "subset": subset, "metric": m, "n_calls": len(g2),
                    "pooled_diff": float(g2["diff"].mean()),
                    "boot_lo": float(lo), "boot_hi": float(hi),
                    "sig": bool(lo > 0 or hi < 0)})
    rt = pd.DataFrame(retest)
    rt.to_csv(out / "beh_diag_retest.csv", index=False)
    print("\nRE-TEST (app_m2 only):")
    print(rt[rt["metric"] == "app_m2"].round(2).to_string(index=False))
    print("\nRE-TEST (speed_m1 only):")
    print(rt[rt["metric"] == "speed_m1"].round(2).to_string(index=False))

    summary = {
        "n_pairs": len(pairs),
        "balance": bal_df.set_index("metric")["smd"].round(3).to_dict(),
        "worst_smd": float(bal_df["smd"].abs().max()),
        "app_m2_during_anogenital": float(
            ctx_df.set_index("behavior").loc["nose2anogenital", "app_m2_in"]),
        "app_m2_elsewhere": float(
            ctx_df.set_index("behavior").loc["nose2anogenital", "app_m2_out"]),
        "retest_app_m2": rt[rt["metric"] == "app_m2"][
            ["matching", "subset", "n_calls", "pooled_diff", "boot_lo",
             "boot_hi", "sig"]].to_dict("records"),
        "retest_speed_m1": rt[rt["metric"] == "speed_m1"][
            ["matching", "subset", "n_calls", "pooled_diff", "boot_lo",
             "boot_hi", "sig"]].to_dict("records"),
    }
    with open(out / "beh_diag_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n" + json.dumps({k: summary[k] for k in
                             ("worst_smd", "app_m2_during_anogenital",
                              "app_m2_elsewhere")}, indent=2))


if __name__ == "__main__":
    main()
