"""Discrete-time hazard models: what triggers a call, and what a call triggers.

Two models with the SAME estimator, run in opposite directions, so the
lead/lag comparison is like-for-like:

  MODEL A (antecedent)  P(call in bin t)
        ~ partner-behaviour onset in (t-1s, t]  +  current context
     "does a recent partner action raise the call hazard, over and above the
      behavioural state the animals are already in?"

  MODEL B (consequence) P(behaviour b starts at t | b was off at t-1)
        ~ any call in (t-1s, t]  +  context measured AT THE START OF THAT
          WINDOW, i.e. BEFORE the call
     "given the same conditions a second earlier, does a call change what
      happens next?"

Model B's controls are deliberately evaluated *before* the call.  That is the
regression form of a matched counterfactual and it includes the outcome's own
baseline - the omission that made the v4 consequence claim regression to the
mean.

INFERENCE.  There are only 12 sessions per genotype, so a cluster-robust
sandwich over a 30-covariate pooled model is rank-deficient (its rank cannot
exceed the number of clusters) and returns meaningless intervals.  Instead
every model is fitted SEPARATELY IN EACH SESSION and the coefficients are
combined across sessions by a t-test on the 12 estimates, with a sign test as
a distribution-free check.  Sessions are the unit of replication, which is
what they actually are.  Headline effects additionally get an exact
circular-shift null of the call train.
"""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import attr_common as A
import attr5_core as C

W = 10                 # 1 s window in 100 ms bins
MIN_EVENTS = 12        # per-session minimum for a usable fit
MIN_SESS = 5
NSHIFT = 500
TEST_BEH = ["nose2anogenital", "nose2body", "approach", "following",
            "chasing", "oriented_toward", "escape", "withdrawal_from_partner",
            "withdrawal_after_contact", "nose2nose", "rearing", "passive"]
COARSE = {
    "investigate": ["nose2anogenital", "nose2body", "following", "chasing",
                    "approach", "oriented_toward"],
    "contact": ["nose2nose", "sidebyside", "sidereside", "fighting"],
    "withdraw": ["withdrawal_from_partner", "escape",
                 "withdrawal_after_contact"],
    "solitary": ["rearing", "passive"],
}


def win_any(v, w=W):
    """1 if v was non-zero anywhere in the w bins ending at t (inclusive)."""
    c = np.concatenate([[0.0], np.cumsum(v)])
    n = v.size
    i = np.arange(n)
    return ((c[i + 1] - c[np.maximum(i + 1 - w, 0)]) > 0).astype(float)


def lag(v, k):
    o = np.full_like(np.asarray(v, float), np.nan)
    if k > 0:
        o[k:] = np.asarray(v, float)[:-k]
    return o


def session_frame(d):
    n = d["n"]
    f = {}
    f["call"] = (d["x_all"] > 0).astype(float)
    for k in ("speed_m1", "speed_m2", "app_m1", "app_m2"):
        f[k] = d[k]
        f[k + "_pre"] = lag(d[k], W)
    f["logdist"] = np.log1p(np.clip(d["dist"], 0, None))
    f["logdist_pre"] = lag(f["logdist"], W)
    for g, fl in COARSE.items():
        v = np.zeros(n)
        for b in fl:
            v = np.maximum(v, d[f"m1_{b}"])
        f[f"st_{g}"] = v
        f[f"st_{g}_pre"] = lag(v, W)
    for who in ("m1", "m2"):
        for b in C.FLAGS:
            f[f"{who}_{b}_lag1"] = lag(d[f"{who}_{b}"], 1)
            f[f"on_{who}_{b}"] = d[f"on_{who}_{b}"]
            f[f"onw_{who}_{b}"] = win_any(d[f"on_{who}_{b}"])
    f["call_prev1"] = win_any(np.concatenate([[0.0], f["call"][:-1]]))
    f["call_prev5"] = win_any(np.concatenate([[0.0], f["call"][:-1]]), 50)
    return pd.DataFrame(f)


def one_fit(df, y, target, ctrl):
    d = df[[y, target] + ctrl].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 500 or d[y].sum() < MIN_EVENTS or d[target].sum() < 5:
        return None
    X = d[[target] + ctrl].to_numpy(float)
    sd = X.std(axis=0)
    keep = sd > 1e-9
    if not keep[0]:
        return None
    X = (X[:, keep] - X[:, keep].mean(axis=0)) / sd[keep]
    m = LogisticRegression(C=10.0, max_iter=2000, solver="lbfgs")
    m.fit(X, d[y].to_numpy(int))
    # rescale the standardised coefficient back to the raw 0/1 target
    return float(m.coef_[0][0] / sd[keep][0])


def combine(betas):
    b = np.array([x for x in betas if x is not None and np.isfinite(x)])
    b = b[np.abs(b) < 8]
    if len(b) < MIN_SESS:
        return None
    t = stats.ttest_1samp(b, 0.0)
    npos = int((b > 0).sum())
    sign_p = stats.binomtest(npos, len(b), 0.5).pvalue
    return {"beta": float(b.mean()), "se": float(b.std(ddof=1) / np.sqrt(len(b))),
            "or": float(np.exp(b.mean())),
            "lo": float(np.exp(b.mean() - 2.2 * b.std(ddof=1) / np.sqrt(len(b)))),
            "hi": float(np.exp(b.mean() + 2.2 * b.std(ddof=1) / np.sqrt(len(b)))),
            "p": float(t.pvalue), "n_sessions": int(len(b)),
            "n_positive": npos, "p_sign": float(sign_p)}


CTRL_A = (["speed_m1", "speed_m2", "app_m1", "app_m2", "logdist",
           "call_prev1", "call_prev5"] + [f"st_{g}" for g in COARSE])
CTRL_B = (["speed_m1_pre", "speed_m2_pre", "app_m1_pre", "app_m2_pre",
           "logdist_pre"] + [f"st_{g}_pre" for g in COARSE])


def model_a(frames, tag, rows):
    for who in ("m1", "m2"):
        for b in TEST_BEH:
            v = f"onw_{who}_{b}"
            r = combine([one_fit(f, "call", v, CTRL_A) for f in frames.values()])
            if r:
                rows.append({"model": "A_antecedent", "set": tag, "who": who,
                             "behavior": b, **r})


def model_b(frames, tag, rows):
    for who in ("m1", "m2"):
        for b in TEST_BEH:
            y = f"on_{who}_{b}"
            bet = []
            for f in frames.values():
                risk = f[f[f"{who}_{b}_lag1"] == 0]     # risk set: b was OFF
                bet.append(one_fit(risk, y, "call_prev1", CTRL_B))
            r = combine(bet)
            if r:
                rows.append({"model": "B_consequence", "set": tag, "who": who,
                             "behavior": b, **r})


def shift_null(sess, keep, which, who, b, n=NSHIFT):
    rng = np.random.default_rng(7)
    stat = []
    for _ in range(n):
        bet = []
        for aid in keep:
            d = dict(sess[aid])
            k = int(rng.integers(50, d["n"] - 50))
            d["x_all"] = np.roll(sess[aid]["x_all"], k)
            f = session_frame(d)
            if which == "A":
                bet.append(one_fit(f, "call", f"onw_{who}_{b}", CTRL_A))
            else:
                risk = f[f[f"{who}_{b}_lag1"] == 0]
                bet.append(one_fit(risk, f"on_{who}_{b}", "call_prev1", CTRL_B))
        r = combine(bet)
        if r:
            stat.append(r["beta"])
    return np.array(stat)


def main():
    out = A.ensure_out()
    bins = C.load_bins()
    calls = pd.read_csv(out / "v5_calls.csv", low_memory=False)
    sess = C.session_arrays(bins, calls)
    wt = [k for k, v in sess.items() if v["genotype"] == "WT"]
    het = [k for k, v in sess.items() if v["genotype"] == "HET"]

    print("building per-session frames ...")
    fw = {a: session_frame(sess[a]) for a in wt}
    fh = {a: session_frame(sess[a]) for a in het}

    rows: list[dict] = []
    print("model A (antecedents) ...")
    model_a(fw, "WT", rows)
    model_a(fh, "HET", rows)
    print("model B (consequences) ...")
    model_b(fw, "WT", rows)
    model_b(fh, "HET", rows)

    print("per call type ...")
    for tname in calls["ctype_name"].dropna().unique():
        sub = calls[calls["ctype_name"] == tname]
        if len(sub) < 80:
            continue
        s2 = C.session_arrays(bins, sub)
        keep = [k for k in wt if k in s2 and s2[k]["x_all"].sum() >= MIN_EVENTS]
        if len(keep) < MIN_SESS:
            continue
        f2 = {a: session_frame(s2[a]) for a in keep}
        model_b(f2, f"type::{tname}", rows)
        model_a(f2, f"type::{tname}", rows)

    df = pd.DataFrame(rows)
    for (m, s), g in df.groupby(["model", "set"]):
        df.loc[g.index, "q"] = C.bh(g["p"].to_numpy())
    df.to_csv(out / "v5_hazard.csv", index=False)

    cols = ["who", "behavior", "or", "lo", "hi", "p", "q", "n_sessions",
            "n_positive"]
    for m in ("A_antecedent", "B_consequence"):
        for s in ("WT", "HET"):
            sub = df[(df["model"] == m) & (df["set"] == s)].sort_values("p")
            if len(sub) == 0:
                continue
            print(f"\n=== {m} ({s}) ===")
            print(sub[cols].to_string(index=False,
                                      float_format=lambda v: f"{v:,.3f}"))

    head = []
    for m, code in (("A_antecedent", "A"), ("B_consequence", "B")):
        sub = df[(df["model"] == m) & (df["set"] == "WT")].sort_values("p")
        if not len(sub):
            continue
        r = sub.iloc[0]
        print(f"\nshift null for {m}: {r['who']}_{r['behavior']} "
              f"(OR={r['or']:.2f}, p={r['p']:.4f}) ...")
        nul = shift_null(sess, wt, code, r["who"], r["behavior"])
        if len(nul):
            p = (np.sum(np.abs(nul - nul.mean())
                        >= abs(r["beta"] - nul.mean())) + 1) / (len(nul) + 1)
            print(f"  observed {r['beta']:+.3f} | null {nul.mean():+.3f} "
                  f"+- {nul.std():.3f} | shift p = {p:.4f}")
            head.append({"model": m, "who": r["who"], "behavior": r["behavior"],
                         "beta": float(r["beta"]), "or": float(r["or"]),
                         "p_session_t": float(r["p"]),
                         "null_mean": float(nul.mean()),
                         "null_sd": float(nul.std()), "p_shift": float(p),
                         "n_draws": int(len(nul))})
    json.dump(head, open(out / "v5_hazard_shiftnull.json", "w"), indent=2)
    print("\nwrote", out / "v5_hazard.csv")


if __name__ == "__main__":
    main()
