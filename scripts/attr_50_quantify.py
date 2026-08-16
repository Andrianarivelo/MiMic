"""Quantification of interaction-window calling, WT vs HET, from the resident's
and the partner's point of view - using DESIGN-BASED attribution bounds.

The acoustic per-call attribution was benchmarked at chance (attr_40): short,
sparse mouse USVs in this dataset carry no measurable individual voice
fingerprint, and received level follows no usable distance law. Per-call
posteriors from the mixture model therefore separate call TYPES (alone-like vs
social-type), not callers, and are exported only as such.

What the DESIGN does support:
1. PARTNER CEILING. Stim animals come from one counterbalanced pool (51557 met
   both a HET and a WT resident). In HET-resident sessions the entire dyad
   emits ~0.7 calls/min - an upper bound on what a stim partner contributes
   when its resident barely engages. Under the stated exchangeability
   assumption (a partner's intrinsic vocal drive does not depend on resident
   genotype except through the interaction itself), the same ceiling applies
   to partner contribution in WT sessions.
2. NATURAL EXPERIMENT. Stim 51558 met two WT residents -> 686 vs 18 dyad
   calls (38x); stim 51557 met a HET and a WT resident -> 24 vs 13. Dyad
   output tracks the resident side, not partner identity.

Bounds per session (interaction window, 300-900 s):
  resident_rate in [dyad_rate - ceiling, dyad_rate]   (clipped at 0)
  partner_rate  in [0, min(ceiling, dyad_rate)]

Outputs under attribution/: session_quantification.csv,
attribution_statistics.csv, timecourse_30s.csv, shared_stim_experiment.csv,
quantification_summary.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A

INTERACTION_MIN = (A.SESSION_END_S - A.PARTNER_INTRO_S) / 60.0
ALONE_MIN = A.PARTNER_INTRO_S / 60.0


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def mw(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, "na"
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(p), stars(p)


def main():
    out = A.ensure_out()
    att = pd.read_csv(out / "call_attribution.csv")
    att["animal_id"] = att["animal_id"].astype(str)
    att = A.apply_genotype(att)

    # ---- per-session base rates ------------------------------------------
    rows = []
    for animal, gt in A.GENOTYPE_MAP.items():
        g = att[att["animal_id"] == animal]
        n_part = int((g["phase"] == "partner").sum())
        n_alone = int((g["phase"] == "alone").sum())
        soc = g[(g["phase"] == "partner")]
        rows.append({
            "animal_id": animal, "genotype": gt,
            "stim_id": A.STIM_MAP.get(animal) or "",
            "n_alone_calls": n_alone, "n_dyad_calls": n_part,
            "alone_rate": n_alone / ALONE_MIN,
            "dyad_rate": n_part / INTERACTION_MIN,
            # call-type split from the mixture model, exported as WHAT IT IS:
            "social_type_frac": float((soc["p_res_hmm"] < 0.5).mean()) if n_part else np.nan,
        })
    ses = pd.DataFrame(rows)

    wt = ses[ses["genotype"] == "WT"]
    het = ses[ses["genotype"] == "HET"]

    # ---- partner ceiling from HET dyads + WT alone (two independent routes)
    ceiling_het_dyad = float(het["dyad_rate"].mean())
    ceiling_het_dyad_p95 = float(het["dyad_rate"].quantile(0.95))
    ceiling_alone = float(wt["alone_rate"].mean())  # a solo animal's typical output
    ceiling = max(ceiling_het_dyad, ceiling_alone)
    ceiling_conservative = max(ceiling_het_dyad_p95, ceiling_alone)

    for cname, ceil in (("", ceiling), ("_cons", ceiling_conservative)):
        ses[f"res_rate_lower{cname}"] = np.maximum(ses["dyad_rate"] - ceil, 0.0)
        ses[f"part_rate_upper{cname}"] = np.minimum(ceil, ses["dyad_rate"])
    ses["res_rate_upper"] = ses["dyad_rate"]
    ses["res_rate_mid"] = (ses["res_rate_lower"] + ses["res_rate_upper"]) / 2
    ses["part_rate_mid"] = ses["part_rate_upper"] / 2
    ses.to_csv(out / "session_quantification.csv", index=False)

    wt = ses[ses["genotype"] == "WT"]
    het = ses[ses["genotype"] == "HET"]

    tests = []
    for metric, label in [
        ("alone_rate", "Alone phase call rate (calls/min)"),
        ("dyad_rate", "Dyad call rate, interaction window (calls/min)"),
        ("res_rate_lower", "Resident-attributed rate, LOWER bound (calls/min)"),
        ("res_rate_upper", "Resident-attributed rate, upper bound = dyad (calls/min)"),
        ("res_rate_lower_cons", "Resident-attributed rate, conservative lower (calls/min)"),
        ("part_rate_upper", "Partner-attributed rate, UPPER bound (calls/min)"),
    ]:
        p, s = mw(wt[metric], het[metric])
        tests.append({
            "metric": metric, "label": label,
            "wt_mean": float(wt[metric].mean()), "wt_median": float(wt[metric].median()),
            "het_mean": float(het[metric].mean()), "het_median": float(het[metric].median()),
            "n_wt": int(wt[metric].notna().sum()), "n_het": int(het[metric].notna().sum()),
            "mannwhitney_p": p, "sig": s,
        })
    tstats = pd.DataFrame(tests)
    tstats.to_csv(out / "attribution_statistics.csv", index=False)
    print(tstats[["label", "wt_mean", "het_mean", "mannwhitney_p", "sig"]].to_string(index=False))

    # ---- 30-s time course -------------------------------------------------
    tc = []
    both = att.copy()
    both["bin"] = (both["start_s"] // 30).astype(int) * 30
    for (animal, b), g in both.groupby(["animal_id", "bin"]):
        tc.append({"animal_id": animal, "genotype": A.GENOTYPE_MAP[str(animal)],
                   "bin_s": int(b), "n_calls": len(g)})
    pd.DataFrame(tc).to_csv(out / "timecourse_30s.csv", index=False)

    # ---- shared-stim natural experiment ----------------------------------
    shared = []
    stim_counts = {}
    for animal, stim in A.STIM_MAP.items():
        if stim:
            stim_counts.setdefault(stim, []).append(animal)
    for stim, animals in sorted(stim_counts.items()):
        for a in animals:
            r = ses[ses["animal_id"] == a].iloc[0]
            shared.append({"stim_id": stim, "n_sessions_for_stim": len(animals),
                           "animal_id": a, "genotype": r["genotype"],
                           "n_dyad_calls": r["n_dyad_calls"],
                           "dyad_rate": r["dyad_rate"]})
    sh = pd.DataFrame(shared)
    sh.to_csv(out / "shared_stim_experiment.csv", index=False)
    reused = sh[sh["n_sessions_for_stim"] >= 2]
    # variance decomposition over reused stims
    nat = {}
    if len(reused):
        grand = reused["dyad_rate"].mean()
        between = reused.groupby("stim_id")["dyad_rate"].mean().sub(grand).pow(2).mean()
        within = reused.groupby("stim_id")["dyad_rate"].var(ddof=0).mean()
        nat = {"reused_stims": reused["stim_id"].nunique(),
               "within_stim_var": float(within), "between_stim_var": float(between),
               "within_frac": float(within / (within + between + 1e-12))}

    summary = {
        "partner_ceiling_calls_per_min": ceiling,
        "partner_ceiling_conservative": ceiling_conservative,
        "ceiling_from_het_dyads_mean": ceiling_het_dyad,
        "ceiling_from_het_dyads_p95": ceiling_het_dyad_p95,
        "ceiling_from_wt_alone_mean": ceiling_alone,
        "wt_resident_share_lower": float(
            1.0 - ceiling / max(wt["dyad_rate"].mean(), 1e-9)),
        "natural_experiment": nat,
        "exchangeability_assumption": (
            "a stim partner's intrinsic vocal drive does not depend on resident "
            "genotype except through the interaction itself; supported by "
            "counterbalanced stim use (51557: HET+WT) and by the shared-stim "
            "variance decomposition"),
    }
    with open(out / "quantification_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
