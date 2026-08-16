"""Supplementary quantification: the intensity ladder and its robustness.

(a) Within-state timing: is calling locked to the ONSET of a behaviour (a
    phasic response) or spread evenly through it (a state readout)?
(b) Bout escalation: HET bouts are nearly all singletons - quantify it.
(c) Leave-one-session-out robustness of the WT social gain (one animal
    contributes 39% of all WT calls, so this must be checked explicitly).
(d) The intensity ladder: call rate along an ordered scale of social
    engagement, per genotype.
"""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import attr_common as A
import attr5_core as C

BIN_MIN = C.BIN_S / 60.0
BOUT_GAP_S = 1.44
RNG = np.random.default_rng(31)
LADDER = ["none", "oriented_toward", "approach", "nose2nose", "nose2body",
          "nose2anogenital", "following", "chasing"]
INVEST = ["nose2anogenital", "nose2body", "following", "chasing", "approach",
          "oriented_toward"]


def main():
    out = A.ensure_out()
    bins = C.load_bins()
    bins["animal_id"] = bins["animal_id"].astype(str)   # session keys are str
    calls = pd.read_csv(out / "v5_calls.csv", low_memory=False)
    calls["animal_id"] = calls["animal_id"].astype(str)
    sess = C.session_arrays(bins, calls)
    wt = [k for k, v in sess.items() if v["genotype"] == "WT"]
    het = [k for k, v in sess.items() if v["genotype"] == "HET"]
    res = {}

    # ---- (a) within-state timing ----------------------------------------
    print("=== (a) are calls locked to behaviour ONSET or spread through it? ===")
    MAXT = 30           # 3 s of elapsed bout time
    prof = np.zeros(MAXT)
    expo = np.zeros(MAXT)
    for aid in wt:
        d = sess[aid]
        inv = np.zeros(d["n"])
        for f in INVEST:
            inv = np.maximum(inv, d[f"m1_{f}"])
        on = C.onset_train(inv)
        x = d["x_all"]
        starts = np.flatnonzero(on)
        for s in starts:
            for k in range(MAXT):
                if s + k >= d["n"] or inv[s + k] < 0.5:
                    break
                expo[k] += 1
                prof[k] += x[s + k]
    rate = prof / np.maximum(expo, 1) / BIN_MIN
    early, late = rate[:5].mean(), rate[10:].mean()
    # trend test on the per-bin rates weighted by exposure
    tt = stats.spearmanr(np.arange(MAXT), rate)
    print(f"  call rate 0-0.5 s into a bout: {early:.1f}/min; "
          f">1 s into it: {late:.1f}/min  (ratio {early/max(late,1e-9):.2f}x)")
    print(f"  Spearman(elapsed, rate) = {tt.statistic:+.2f} (p={tt.pvalue:.3f})"
          f"   -> flat means calls index the STATE, not its onset")
    res["timing"] = {"rate_early": float(early), "rate_late": float(late),
                     "rho": float(tt.statistic), "p": float(tt.pvalue),
                     "profile": rate.tolist(),
                     "exposure": expo.tolist()}
    pd.DataFrame({"elapsed_s": np.arange(MAXT) * C.BIN_S, "rate": rate,
                  "exposure_bins": expo}).to_csv(
        out / "v5_bout_timing.csv", index=False)

    # ---- (b) bout escalation --------------------------------------------
    print("\n=== (b) escalation within a calling bout ===")
    rows = []
    for (aid, gt), g in calls.groupby(["animal_id", "genotype"]):
        t = np.sort(g["start_s"].to_numpy())
        newb = np.concatenate([[True], np.diff(t) > BOUT_GAP_S])
        bid = np.cumsum(newb) - 1
        sizes = np.bincount(bid)
        rows.append({"animal_id": aid, "genotype": gt, "n_bouts": len(sizes),
                     "p_ge2": float(np.mean(sizes >= 2)),
                     "p_ge4": float(np.mean(sizes >= 4)),
                     "mean_size": float(sizes.mean())})
    bo = pd.DataFrame(rows)
    for col in ("p_ge2", "p_ge4"):
        w = bo.loc[bo.genotype == "WT", col]
        h = bo.loc[bo.genotype == "HET", col]
        u = stats.mannwhitneyu(w, h)
        print(f"  P(bout has >= {col[-1]} calls): WT {w.median():.3f}  "
              f"HET {h.median():.3f}   p = {u.pvalue:.4f}")
        res.setdefault("escalation_bout", {})[col] = {
            "WT": float(w.median()), "HET": float(h.median()),
            "p": float(u.pvalue)}
    bo.to_csv(out / "v5_bout_escalation.csv", index=False)

    # ---- (c) LOSO robustness of the social gain -------------------------
    print("\n=== (c) leave-one-session-out: WT social gain ===")
    bins2 = bins.copy()
    soc = np.zeros(len(bins2), bool)
    for f in INVEST + ["sidebyside", "sidereside", "fighting"]:
        soc |= bins2[f"m1_{f}"].to_numpy() > 0.5
    bins2["soc"] = soc
    nonsoc = np.ones(len(bins2), bool)
    for f in C.FLAGS:
        nonsoc &= bins2[f"m1_{f}"].to_numpy() < 0.5
    bins2["nonsoc"] = nonsoc

    def gain(ids):
        d = bins2[bins2["animal_id"].isin(ids)]
        a = d.loc[d.soc, "n_calls"].sum() / max(d.soc.sum() * BIN_MIN, 1e-9)
        b = d.loc[d.nonsoc, "n_calls"].sum() / max(d.nonsoc.sum() * BIN_MIN, 1e-9)
        return a / max(b, 1e-9)

    full = gain(wt)
    loso = {a: gain([x for x in wt if x != a]) for a in wt}
    lo, hi = min(loso.values()), max(loso.values())
    print(f"  full WT gain {full:.2f}x; leave-one-out range {lo:.2f}-{hi:.2f}x")
    worst = min(loso, key=lambda k: loso[k])
    print(f"  dropping the biggest caller ({worst}) gives {loso[worst]:.2f}x")
    print(f"  HET gain {gain(het):.2f}x")
    res["loso_gain"] = {"full": float(full), "min": float(lo),
                        "max": float(hi), "worst_session": worst,
                        "HET": float(gain(het)),
                        "per_session": {k: float(v) for k, v in loso.items()}}

    # ---- (d) intensity ladder -------------------------------------------
    print("\n=== (d) the intensity ladder ===")
    lad = []
    for gt, ids in (("WT", wt), ("HET", het)):
        d = bins2[bins2["animal_id"].isin(ids)]
        for f in LADDER:
            m = (d["m1_" + f] > 0.5) if f != "none" else d["nonsoc"]
            n = int(m.sum())
            if n < 100:
                continue
            per = [d[d.animal_id == a].loc[m[d.animal_id == a], "n_calls"].sum()
                   / max((m[d.animal_id == a]).sum() * BIN_MIN, 1e-9)
                   for a in ids]
            per = [p for p in per if np.isfinite(p)]
            lad.append({"genotype": gt, "state": f, "n_bins": n,
                        "rate": d.loc[m, "n_calls"].sum() / (n * BIN_MIN),
                        "median_session_rate": float(np.median(per)),
                        "n_calls": float(d.loc[m, "n_calls"].sum())})
    ladder = pd.DataFrame(lad)
    if ladder.empty:
        raise SystemExit("ladder empty - check animal_id dtypes")
    ladder.to_csv(out / "v5_ladder.csv", index=False)
    piv = ladder.pivot(index="state", columns="genotype", values="rate")
    piv = piv.reindex([s for s in LADDER if s in piv.index])
    piv["ratio"] = piv["WT"] / piv["HET"].clip(lower=1e-9)
    print(piv.round(2).to_string())
    for gt in ("WT", "HET"):
        sub = ladder[ladder.genotype == gt].set_index("state").reindex(
            [s for s in LADDER if s in set(ladder.state)])
        r = stats.spearmanr(np.arange(len(sub)), sub["rate"])
        print(f"  {gt}: Spearman(ladder position, call rate) = "
              f"{r.statistic:+.2f} (p={r.pvalue:.3f})")
        res.setdefault("ladder", {})[gt] = {"rho": float(r.statistic),
                                            "p": float(r.pvalue)}
    json.dump(res, open(out / "v5_supplement.json", "w"), indent=2, default=float)
    print("\nwrote", out / "v5_supplement.json")


if __name__ == "__main__":
    main()
