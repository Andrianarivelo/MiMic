"""Who leads whom: calls vs behaviour, by the antisymmetric cross-correlogram.

We compute the CALL-TRIGGERED BEHAVIOUR PROFILE

    P(tau) = ( sum_t x(t) y(t + tau) ) / sum_t x(t)

i.e. the probability that behaviour `y` is occupied tau seconds after a call
(tau < 0 = before), and split it into even and odd parts.  By the argument in
attr5_core, an instantaneous common drive -- proximity, arousal, bout
structure, regression to the mean -- contributes only to the EVEN part, so

    Lambda = sum_{tau > 0} [ P(tau) - P(-tau) ]

    Lambda > 0  behaviour is more likely AFTER the call  -> call LEADS
    Lambda < 0  behaviour is more likely BEFORE the call -> behaviour LEADS

IMPORTANT: `y` must be the STATE INDICATOR, not a bout-onset train.  An onset
train is asymmetric by construction (the state is off before the onset and on
after it), so calls that merely occur *during* a behaviour would manufacture a
negative Lambda with no directional coupling whatsoever.  With the state
indicator the common-drive term is a genuine autocovariance and cancels.
The onset-locked rate profiles are still saved, but only as description.
"""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import attr_common as A
import attr5_core as C

LMAX = 50          # +-5 s
NDRAW = 6000


def profile_block(sess, keep, xkey, ykey, lmax=LMAX, n_draw=NDRAW):
    """Call-triggered behaviour profile pooled over sessions, with the exact
    circular-shift null of the call train."""
    num_pos = np.zeros(lmax)
    num_neg = np.zeros(lmax)
    denom = 0.0
    nulls_pos, nulls_neg = [], []
    used = []
    for aid in keep:
        d = sess[aid]
        x, y = d[xkey], d[ykey]
        if x.sum() < 5 or y.sum() < 10:
            continue
        p, q = C.odd_profile(x, y, lmax)
        num_pos += p
        num_neg += q
        denom += x.sum()
        used.append(aid)
        np_, nn_ = C.exact_shift_null(x, y, lmax)
        sel = C.RNG.integers(0, np_.shape[0], size=n_draw)
        nulls_pos.append(np_[sel])
        nulls_neg.append(nn_[sel])
    if denom < 25 or len(used) < 3:
        return None
    NP, NN = np.sum(nulls_pos, axis=0), np.sum(nulls_neg, axis=0)
    after = num_pos / denom          # P(behaviour | tau > 0)
    before = num_neg / denom         # P(behaviour | tau < 0)
    odd_obs = (num_pos - num_neg) / denom
    odd_null = (NP - NN) / denom
    lam = float(odd_obs.sum())
    lam_null = odd_null.sum(axis=1)
    p_two = (np.sum(np.abs(lam_null - lam_null.mean())
                    >= abs(lam - lam_null.mean())) + 1) / (len(lam_null) + 1)
    # latency: lag of the largest |odd| excursion
    j = int(np.argmax(np.abs(odd_obs)))
    return {
        "lags": np.arange(1, lmax + 1) * C.BIN_S,
        "after": after, "before": before,
        "even": 0.5 * (after + before),
        "even_null": 0.5 * (NP + NN).mean(axis=0) / denom,
        "odd": odd_obs,
        "odd_lo": np.percentile(odd_null, 2.5, axis=0),
        "odd_hi": np.percentile(odd_null, 97.5, axis=0),
        "lambda": lam,
        "z": float((lam - lam_null.mean()) / (lam_null.std() + 1e-12)),
        "p": float(p_two),
        "lam_null_mean": float(lam_null.mean()),
        "lam_null_sd": float(lam_null.std()),
        "latency_s": float((j + 1) * C.BIN_S * np.sign(odd_obs[j])),
        "n_calls": float(denom), "n_sessions": len(used),
        "occupancy": float(np.mean([sess[a][ykey].mean() for a in used])),
    }


def run_block(sess, keep, xkey, tag, rows, profiles):
    for who in ("m1", "m2"):
        for f in C.FLAGS:
            if who == "m2" and f in C.MUTUAL:
                continue                      # identical to the m1 copy
            r = profile_block(sess, keep, xkey, f"{who}_{f}")
            if r is None:
                continue
            rows.append({
                "set": tag, "who": who, "behavior": f,
                "n_calls": r["n_calls"], "n_sessions": r["n_sessions"],
                "occupancy": r["occupancy"],
                "p_before": float(r["before"][:20].mean()),
                "p_after": float(r["after"][:20].mean()),
                "enrich_even": float(r["even"][:20].mean()
                                     / max(r["even_null"][:20].mean(), 1e-9)),
                "lambda": r["lambda"], "z": r["z"], "p": r["p"],
                "latency_s": r["latency_s"],
                "directional": f in C.DIRECTIONAL,
            })
            profiles[f"{tag}|{who}|{f}"] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in r.items()}


def main():
    out = A.ensure_out()
    bins = C.load_bins()
    calls = pd.read_csv(out / "v5_calls.csv", low_memory=False)
    sess = C.session_arrays(bins, calls)
    wt = [k for k, v in sess.items() if v["genotype"] == "WT"]
    het = [k for k, v in sess.items() if v["genotype"] == "HET"]
    print(f"{len(wt)} WT / {len(het)} HET sessions")

    rows: list[dict] = []
    profiles: dict = {}
    run_block(sess, wt, "x_res", "WT", rows, profiles)
    run_block(sess, het, "x_res", "HET", rows, profiles)
    run_block(sess, wt + het, "x_res", "ALL", rows, profiles)

    for tname in calls["ctype_name"].dropna().unique():
        sub = calls[calls["ctype_name"] == tname]
        if len(sub) < 80:
            continue
        s2 = C.session_arrays(bins, sub)
        keep = [k for k in wt if k in s2 and s2[k]["x_res"].sum() >= 10]
        if len(keep) >= 4:
            run_block(s2, keep, "x_res", f"type::{tname}", rows, profiles)

    df = pd.DataFrame(rows)
    for tag, g in df.groupby("set"):
        df.loc[g.index, "q"] = C.bh(g["p"].to_numpy())
    df.to_csv(out / "v5_directionality.csv", index=False)
    json.dump(profiles, open(out / "v5_direction_profiles.json", "w"))

    m = df[df["set"] == "WT"].sort_values("lambda")
    print("\n=== WT: call-triggered behaviour profile ===")
    print("  lambda < 0 : behaviour LEADS the call (call is a response)")
    print("  lambda > 0 : call LEADS the behaviour (call is a trigger)\n")
    cols = ["who", "behavior", "occupancy", "enrich_even", "p_before",
            "p_after", "lambda", "z", "p", "q", "latency_s"]
    print(m[cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"\n{(m['q'] < 0.05).sum()} of {len(m)} survive FDR")
    print("wrote", out / "v5_directionality.csv")


if __name__ == "__main__":
    main()
