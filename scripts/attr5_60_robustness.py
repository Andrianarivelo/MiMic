"""Is the gain result carried by one animal?

31078 contributes 703 of the 1,786 WT interaction-window calls (39%).  Every
headline number is therefore recomputed (a) with that animal dropped and (b) as
a PER-SESSION statistic, where each animal contributes one value regardless of
how loud it is.  The per-session form is the decisive one: the social gain is a
WITHIN-animal ratio, so it cannot be inflated by an animal that simply calls a
lot.
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
from attr5_30_why_het import kitagawa, state_of, distance_gain

BIN_MIN = C.BIN_S / 60.0
RNG = np.random.default_rng(97)
BIG = "31078"
INVEST = ["nose2anogenital", "nose2body", "following", "chasing", "approach",
          "oriented_toward"]
LADDER = ["none", "oriented_toward", "approach", "nose2nose", "nose2body",
          "nose2anogenital", "following", "chasing"]


def main():
    out = A.ensure_out()
    bins = C.load_bins()
    bins["animal_id"] = bins["animal_id"].astype(str)
    bins["state"] = state_of(bins)
    soc = bins["state"].isin(["investigate", "contact"])
    bins["soc"] = soc
    bins["non"] = bins["state"] == "none"
    ids = {g: sorted(bins.loc[bins.genotype == g, "animal_id"].unique())
           for g in ("WT", "HET")}
    res = {}

    # ---------- 1. per-session social gain (one value per animal) ----------
    print("=== 1. social gain computed PER SESSION (tail-immune) ===")
    rows = []
    for gt, aa in ids.items():
        for a in aa:
            d = bins[bins.animal_id == a]
            cs, ts = d.loc[d.soc, "n_calls"].sum(), d.soc.sum() * BIN_MIN
            cn, tn = d.loc[d.non, "n_calls"].sum(), d.non.sum() * BIN_MIN
            # Haldane-style smoothing so zero-call cells stay finite
            gain = ((cs + 0.5) / ts) / ((cn + 0.5) / tn)
            rows.append({"animal_id": a, "genotype": gt, "gain": gain,
                         "calls": int(d["n_calls"].sum()),
                         "rate_soc": cs / ts, "rate_non": cn / tn})
    per = pd.DataFrame(rows).sort_values(["genotype", "gain"])
    print(per.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    w = per[per.genotype == "WT"]["gain"]
    h = per[per.genotype == "HET"]["gain"]
    u = stats.mannwhitneyu(w, h)
    print(f"\n  WT  median gain {w.median():.2f}  ({(w > 1).sum()}/12 sessions > 1)")
    print(f"  HET median gain {h.median():.2f}  ({(h > 1).sum()}/12 sessions > 1)")
    print(f"  Mann-Whitney p = {u.pvalue:.4f}   (rank test: 31078 counts once)")
    wn = per[(per.genotype == "WT") & (per.animal_id != BIG)]["gain"]
    un = stats.mannwhitneyu(wn, h)
    print(f"  WITHOUT {BIG}: WT median {wn.median():.2f} "
          f"({(wn > 1).sum()}/11 > 1), p = {un.pvalue:.4f}")
    print(f"  31078's own gain = {per.loc[per.animal_id == BIG, 'gain'].iloc[0]:.2f} "
          f"(rank {int((w < per.loc[per.animal_id == BIG, 'gain'].iloc[0]).sum()) + 1} of 12)")
    res["per_session_gain"] = {
        "WT_median": float(w.median()), "HET_median": float(h.median()),
        "WT_n_above_1": int((w > 1).sum()), "HET_n_above_1": int((h > 1).sum()),
        "p": float(u.pvalue),
        "WT_median_drop31078": float(wn.median()),
        "WT_n_above_1_drop31078": int((wn > 1).sum()),
        "p_drop31078": float(un.pvalue),
        "gain_31078": float(per.loc[per.animal_id == BIG, "gain"].iloc[0])}
    per.to_csv(out / "v5_gain_per_session.csv", index=False)

    # ---------- 2. pooled gain with and without 31078 ----------------------
    print("\n=== 2. pooled (person-time) social gain ===")

    def pooled_gain(sel):
        d = bins[bins.animal_id.isin(sel)]
        a = d.loc[d.soc, "n_calls"].sum() / max(d.soc.sum() * BIN_MIN, 1e-9)
        b = d.loc[d.non, "n_calls"].sum() / max(d.non.sum() * BIN_MIN, 1e-9)
        return a / max(b, 1e-9)

    wt_no = [a for a in ids["WT"] if a != BIG]
    for lab, sel in (("WT all 12", ids["WT"]), (f"WT without {BIG}", wt_no),
                     ("HET all 12", ids["HET"])):
        boot = [pooled_gain(list(RNG.choice(sel, len(sel), replace=True)))
                for _ in range(4000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {lab:<20} {pooled_gain(sel):5.2f}x  [{lo:.2f}, {hi:.2f}]")
        res.setdefault("pooled_gain", {})[lab] = {
            "gain": float(pooled_gain(sel)), "ci": [float(lo), float(hi)]}
    bw = [pooled_gain(list(RNG.choice(wt_no, len(wt_no), replace=True)))
          for _ in range(4000)]
    bh = [pooled_gain(list(RNG.choice(ids["HET"], 12, replace=True)))
          for _ in range(4000)]
    ratio = np.array(bw) / np.maximum(bh, 1e-9)
    p = 2 * min(np.mean(ratio <= 1), np.mean(ratio >= 1))
    print(f"  ratio WT(no {BIG}) / HET = {pooled_gain(wt_no)/pooled_gain(ids['HET']):.2f}x "
          f"[{np.percentile(ratio,2.5):.2f}, {np.percentile(ratio,97.5):.2f}], p = {p:.4f}")
    res["gain_ratio_drop31078"] = {
        "ratio": float(pooled_gain(wt_no) / pooled_gain(ids["HET"])),
        "ci": [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))],
        "p": float(p)}

    # ---------- 3. the state-specific ladder of deficits -------------------
    print("\n=== 3. is the deficit uniform across states? ===")
    print("    (a uniform 'volume knob' would give the SAME ratio in every row)")
    for lab, sel in (("all 12 WT", ids["WT"]), (f"WT minus {BIG}", wt_no)):
        print(f"\n  -- {lab} --")
        rows2 = []
        for s in ("none", "solitary", "withdraw", "investigate", "contact"):
            dw = bins[bins.animal_id.isin(sel) & (bins.state == s)]
            dh = bins[bins.animal_id.isin(ids["HET"]) & (bins.state == s)]
            rw = dw["n_calls"].sum() / max(len(dw) * BIN_MIN, 1e-9)
            rh = dh["n_calls"].sum() / max(len(dh) * BIN_MIN, 1e-9)
            rows2.append({"state": s, "WT": rw, "HET": rh, "ratio": rw / max(rh, 1e-9)})
            print(f"    {s:>12}: WT {rw:6.2f}  HET {rh:5.2f}  ->  {rw/max(rh,1e-9):5.1f}x")
        res.setdefault("state_ratios", {})[lab] = rows2

    # ---------- 4. everything else, dropping 31078 ------------------------
    print("\n=== 4. the other headline numbers without 31078 ===")
    sub = bins[bins.animal_id != BIG]
    kk = kitagawa(sub, n_boot=2000)
    print(f"  Kitagawa propensity share {kk['pct_propensity']:.1f}% "
          f"[{kk['pct_propensity_ci'][0]:.1f}, {kk['pct_propensity_ci'][1]:.1f}]"
          f"   (all 12: 90.3%)")
    print(f"  counterfactual HET on WT's time budget: "
          f"{kk['cf_HET_with_WT_time']:.2f}/min vs {kk['R_HET']:.2f} observed")
    res["kitagawa_drop31078"] = {
        "pct_propensity": kk["pct_propensity"],
        "ci": kk["pct_propensity_ci"],
        "R_WT": kk["R_WT"], "R_HET": kk["R_HET"],
        "cf": kk["cf_HET_with_WT_time"]}

    # ladder correlation
    nonsoc = np.ones(len(bins), bool)
    for f in C.FLAGS:
        nonsoc &= bins[f"m1_{f}"].to_numpy() < 0.5
    bins["__none"] = nonsoc
    for lab, sel in (("WT all 12", ids["WT"]), (f"WT minus {BIG}", wt_no),
                     ("HET", ids["HET"])):
        d = bins[bins.animal_id.isin(sel)]
        rates = []
        for s in LADDER:
            m = d["__none"] if s == "none" else (d["m1_" + s] > 0.5)
            if m.sum() < 100:
                rates.append(np.nan)
                continue
            rates.append(d.loc[m, "n_calls"].sum() / (m.sum() * BIN_MIN))
        r = stats.spearmanr(np.arange(len(rates)), rates, nan_policy="omit")
        print(f"  ladder Spearman, {lab:<16} rho = {r.statistic:+.2f} "
              f"(p = {r.pvalue:.3f})")
        res.setdefault("ladder", {})[lab] = {"rho": float(r.statistic),
                                            "p": float(r.pvalue)}

    # distance slope
    def slope(sel):
        d = bins[bins.animal_id.isin(sel)].dropna(subset=["dist"])
        e = np.nanpercentile(bins["dist"].dropna(), np.linspace(0, 100, 9))
        e[0], e[-1] = -np.inf, np.inf
        q = pd.cut(d["dist"], e, labels=False, duplicates="drop")
        x, y = [], []
        for _, g in d.groupby(q):
            if len(g) < 200:
                continue
            x.append(np.log(g["dist"].median()))
            y.append(np.log((g["n_calls"].sum() + 0.5) / (len(g) * BIN_MIN)))
        return float(np.polyfit(x, y, 1)[0])
    print(f"  distance slope, WT all 12  {slope(ids['WT']):+.2f}")
    print(f"  distance slope, WT minus {BIG}  {slope(wt_no):+.2f}"
          f"   (HET {slope(ids['HET']):+.2f})")
    res["distance_slope_drop31078"] = float(slope(wt_no))

    # overall rate comparison, rank based
    rate = bins.groupby(["animal_id", "genotype"])["n_calls"].sum().reset_index()
    rate["rate"] = rate["n_calls"] / 10.0
    rw = rate[(rate.genotype == "WT") & (rate.animal_id != BIG)]["rate"]
    rh = rate[rate.genotype == "HET"]["rate"]
    uu = stats.mannwhitneyu(rw, rh)
    print(f"\n  session call rate WITHOUT {BIG}: WT median {rw.median():.1f}/min "
          f"vs HET {rh.median():.1f}/min, Mann-Whitney p = {uu.pvalue:.4f}")
    res["rate_drop31078"] = {"WT_median": float(rw.median()),
                             "HET_median": float(rh.median()),
                             "p": float(uu.pvalue)}

    json.dump(res, open(out / "v5_robustness.json", "w"), indent=2, default=float)
    print("\nwrote", out / "v5_robustness.json")


if __name__ == "__main__":
    main()
