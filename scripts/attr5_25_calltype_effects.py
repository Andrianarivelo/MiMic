"""Do particular call types do particular things?  A within-call design.

The v4 consequence claim failed because it compared calls with non-calls, and
calls are emitted at kinematic extremes; the contrast was contaminated by
regression to the mean.  The repair here is to keep the comparison ENTIRELY
WITHIN CALLS: every arm is a call, so whatever selects a moment for calling is
shared by both arms and cancels.  Only the call's TYPE differs.

For every call we take a pre-window [-1, 0) s and a post-window (0, +1] s, and
model

    Delta(outcome) ~ call type + pre-window value of that same outcome
                     + pre-window context + session fixed effects

Conditioning on the pre-window value of the outcome is the point: it is the
control whose absence sank v4.

Null: call-type labels are permuted BETWEEN BOUTS within a session, preserving
each session's type mix, each bout's internal composition and all call times.
Under that null a type label carries no information, which is exactly the
hypothesis being tested.
"""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import attr_common as A
import attr5_core as C

PRE, POST = 10, 10          # 1 s each side, in 100 ms bins
BOUT_GAP_S = 1.44           # from the ICI mixture in attr5_30
NPERM = 1000
OUTCOMES = ["app_m2", "app_m1", "speed_m2", "speed_m1", "dist"]
RNG = np.random.default_rng(23)


def build(sess, calls, keep):
    """One row per call: pre/post windows for each outcome, plus context."""
    rows = []
    for aid in keep:
        d = sess[aid]
        n = d["n"]
        c = calls[calls["animal_id"].astype(str) == str(aid)].sort_values("start_s")
        if len(c) < 5:
            continue
        t = c["start_s"].to_numpy()
        bout = np.concatenate([[0], np.cumsum(np.diff(t) > BOUT_GAP_S)])
        idx = np.clip(((t - C.T0) / C.BIN_S).astype(int), PRE, n - POST - 1)
        # escape-onset outcome
        esc = d["on_m2_escape"]
        wdr = d["on_m2_withdrawal_from_partner"]
        for j, (i, ci) in enumerate(zip(idx, c.index)):
            r = {"sid": aid, "call_id": c.loc[ci, "call_id"],
                 "ctype": c.loc[ci, "ctype_name"], "bout": int(bout[j]),
                 "t": t[j], "dur_ms": c.loc[ci, "rf_dur_ms"],
                 "amp": c.loc[ci, "rf_amp_peak_db"],
                 "n_in_bout": int(np.sum(bout == bout[j]))}
            for o in OUTCOMES:
                v = d[o]
                pre = np.nanmean(v[i - PRE:i])
                post = np.nanmean(v[i + 1:i + 1 + POST])
                r[f"pre_{o}"] = pre
                r[f"post_{o}"] = post
                r[f"d_{o}"] = post - pre
            r["esc_post"] = float(esc[i + 1:i + 1 + 2 * POST].sum() > 0)
            r["esc_pre"] = float(esc[i - 2 * PRE:i].sum() > 0)
            r["wdr_post"] = float(wdr[i + 1:i + 1 + 2 * POST].sum() > 0)
            r["wdr_pre"] = float(wdr[i - 2 * PRE:i].sum() > 0)
            for f in ("nose2anogenital", "following", "chasing", "approach"):
                r[f"pre_m1_{f}"] = float(d[f"m1_{f}"][i - PRE:i].mean())
            rows.append(r)
    return pd.DataFrame(rows)


CTRL = ([f"pre_{o}" for o in OUTCOMES]
        + ["pre_m1_nose2anogenital", "pre_m1_following", "pre_m1_chasing",
           "pre_m1_approach"])
MIN_PER_ARM = 8
MIN_SESS = 5


def _session_beta(d, outcome, group_col, ref, alt):
    """Adjusted alt-minus-ref difference inside ONE session."""
    if (d[group_col] == alt).sum() < MIN_PER_ARM or \
       (d[group_col] == ref).sum() < MIN_PER_ARM:
        return None
    X = pd.DataFrame({"alt": (d[group_col] == alt).astype(float)})
    for c_ in CTRL:
        if c_ in d.columns:
            X[c_] = d[c_].astype(float)
    X = X.fillna(X.mean())
    X = X.loc[:, X.std() > 1e-9]
    if "alt" not in X.columns or len(X) < 25:
        return None
    X = sm.add_constant(X, has_constant="add")
    try:
        m = sm.OLS(d[outcome].to_numpy(float), X.to_numpy(float)).fit()
    except Exception:                                   # noqa: BLE001
        return None
    return float(m.params[list(X.columns).index("alt")])


def adjusted_effect(df, outcome, group_col, ref, alt):
    """Per-session adjusted differences, combined across sessions by a t-test.

    With 12 sessions a pooled cluster-robust sandwich over ~20 covariates is
    rank-deficient, so sessions are treated as what they are: replicates.
    """
    d = df[df[group_col].isin([ref, alt])].dropna(subset=[outcome])
    betas = [_session_beta(g, outcome, group_col, ref, alt)
             for _, g in d.groupby("sid")]
    b = np.array([x for x in betas if x is not None and np.isfinite(x)])
    if len(b) < MIN_SESS:
        return None
    t = stats.ttest_1samp(b, 0.0)
    return {"beta": float(b.mean()),
            "se": float(b.std(ddof=1) / np.sqrt(len(b))),
            "p": float(t.pvalue), "n": int(len(d)),
            "n_sessions": int(len(b)), "n_positive": int((b > 0).sum()),
            "n_alt": int((d[group_col] == alt).sum()),
            "n_ref": int((d[group_col] == ref).sum())}


def perm_labels(df, rng):
    """Permute type labels between bouts within each session."""
    out = df.copy()
    for sid, g in df.groupby("sid"):
        bt = g.groupby("bout")["ctype"].first()
        shuffled = rng.permutation(bt.to_numpy())
        mapping = dict(zip(bt.index, shuffled))
        out.loc[g.index, "ctype"] = g["bout"].map(mapping).to_numpy()
    return out


def main():
    out = A.ensure_out()
    bins = C.load_bins()
    calls = pd.read_csv(out / "v5_calls.csv", low_memory=False)
    sess = C.session_arrays(bins, calls)
    wt = [k for k, v in sess.items() if v["genotype"] == "WT"]
    df = build(sess, calls, wt)
    df = df.dropna(subset=["ctype"])
    print(f"{len(df)} WT calls with complete windows, "
          f"{df['sid'].nunique()} sessions, {df['bout'].nunique()} bout ids")
    print(df["ctype"].value_counts().to_string())
    df.to_csv(out / "v5_calltype_windows.csv", index=False)

    # collapse to the contrast with the most biological content:
    # sustained (long/very long flat) vs brief (ultrashort) calls
    df["dur_class"] = np.where(df["dur_ms"] < 9, "ultrashort",
                               np.where(df["dur_ms"] >= 20, "long", "short"))
    print("\nduration classes:", df["dur_class"].value_counts().to_dict())

    rows = []
    tests = [("dur_class", "ultrashort", "long")]
    types = [t for t in df["ctype"].value_counts().index
             if df["ctype"].value_counts()[t] >= 60]
    ref_type = df["ctype"].value_counts().index[0]
    tests += [("ctype", ref_type, t) for t in types if t != ref_type]

    outcomes = ["d_app_m2", "d_app_m1", "d_speed_m2", "d_speed_m1", "d_dist",
                "esc_post", "wdr_post"]
    for col, ref, alt in tests:
        for o in outcomes:
            r = adjusted_effect(df, o, col, ref, alt)
            if r:
                rows.append({"contrast": f"{alt} vs {ref}", "by": col,
                             "outcome": o, **r})
    res = pd.DataFrame(rows)
    res["q"] = C.bh(res["p"].to_numpy())
    res = res.sort_values("p")
    res.to_csv(out / "v5_calltype_effects.csv", index=False)
    print("\n=== adjusted within-call contrasts (cluster-robust by session) ===")
    print(res[["contrast", "outcome", "n", "beta", "se", "p", "q"]]
          .head(18).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    # ---- omnibus permutation test: does type carry ANY information? ------
    print(f"\n=== block permutation of type labels ({NPERM} draws) ===")
    obs_stats, null_stats = {}, {o: [] for o in outcomes}
    for o in outcomes:
        r = adjusted_effect(df, o, "dur_class", "ultrashort", "long")
        obs_stats[o] = abs(r["beta"] / r["se"]) if r else np.nan
    for _ in range(NPERM):
        p = perm_labels(df, RNG)
        p["dur_class"] = np.where(
            p["ctype"].str.startswith("ultrashort"), "ultrashort",
            np.where(p["ctype"].str.startswith(("long", "very long")),
                     "long", "short"))
        for o in outcomes:
            r = adjusted_effect(p, o, "dur_class", "ultrashort", "long")
            null_stats[o].append(abs(r["beta"] / r["se"]) if r else np.nan)
    perm = []
    for o in outcomes:
        nul = np.array(null_stats[o], float)
        nul = nul[np.isfinite(nul)]
        pv = (np.sum(nul >= obs_stats[o]) + 1) / (len(nul) + 1)
        perm.append({"outcome": o, "obs_t": obs_stats[o],
                     "null_mean_t": float(np.mean(nul)), "p_perm": float(pv),
                     "n_draws": int(len(nul))})
        print(f"  {o:>12}: |t|={obs_stats[o]:5.2f}  null mean |t|="
              f"{np.mean(nul):4.2f}   permutation p = {pv:.4f}")
    pdf = pd.DataFrame(perm)
    pdf["q_perm"] = C.bh(pdf["p_perm"].to_numpy())
    pdf.to_csv(out / "v5_calltype_permutation.csv", index=False)

    # ---- dose-response: does bout size predict the post-call change? -----
    print("\n=== dose-response (calls in bout), adjusted for pre-window ===")
    dose = []
    df["log_dose"] = np.log(df["n_in_bout"])
    for o in outcomes:
        betas = []
        for _, d in df.dropna(subset=[o]).groupby("sid"):
            if len(d) < 25 or d["log_dose"].std() < 1e-9:
                continue
            X = pd.DataFrame({"dose": d["log_dose"].astype(float)})
            for c_ in [f"pre_{k}" for k in OUTCOMES]:
                X[c_] = d[c_].astype(float)
            X = X.fillna(X.mean())
            X = sm.add_constant(X.loc[:, X.std() > 1e-9], has_constant="add")
            if "dose" not in X.columns:
                continue
            m = sm.OLS(d[o].to_numpy(float), X.to_numpy(float)).fit()
            betas.append(float(m.params[list(X.columns).index("dose")]))
        if len(betas) < MIN_SESS:
            continue
        b = np.array(betas)
        t = stats.ttest_1samp(b, 0.0)
        dose.append({"outcome": o, "beta": float(b.mean()),
                     "se": float(b.std(ddof=1) / np.sqrt(len(b))),
                     "p": float(t.pvalue), "n_sessions": int(len(b)),
                     "n": int(len(df))})
        print(f"  {o:>12}: {b.mean():+8.2f} +- "
              f"{b.std(ddof=1)/np.sqrt(len(b)):.2f}  p = {t.pvalue:.3f}  "
              f"({len(b)} sessions)")
    ddf = pd.DataFrame(dose)
    ddf["q"] = C.bh(ddf["p"].to_numpy())
    ddf.to_csv(out / "v5_dose_response.csv", index=False)

    json.dump({"n_calls": int(len(df)),
               "n_sessions": int(df["sid"].nunique()),
               "ref_type": ref_type,
               "any_sig_q05": int((res["q"] < 0.05).sum()),
               "perm_min_q": float(pdf["q_perm"].min()),
               "dose_min_q": float(ddf["q"].min())},
              open(out / "v5_calltype_summary.json", "w"), indent=2)
    print("\nwrote", out / "v5_calltype_effects.csv")


if __name__ == "__main__":
    main()
