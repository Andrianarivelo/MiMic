"""V2 final quantification from the GEO attribution (social-geometry cue,
session prevalence MLE with profile-likelihood CIs).

Resident rate per session = pi_hat * n / 10 min, CI propagated from the
prevalence CI. Partner rate is the complement. WT vs HET, both perspectives.

Outputs: v2_session_quantification.csv, v2_attribution_statistics.csv,
v2_timecourse.csv, v2_quant_summary.json
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
    ses = pd.read_csv(out / "v2_geo_sessions.csv")
    ses["animal_id"] = ses["animal_id"].astype(str)
    att = pd.read_csv(out / "v2_geo_attribution.csv")
    att["animal_id"] = att["animal_id"].astype(str)
    calls = A.load_calls()

    alone_n = calls[calls["phase"] == "alone"].groupby("animal_id").size()
    ses["n_alone"] = ses["animal_id"].map(alone_n).fillna(0).astype(int)
    ses["alone_rate"] = ses["n_alone"] / ALONE_MIN
    ses["dyad_rate"] = ses["n"] / INTERACTION_MIN
    for c, pcol in (("res_rate", "pi_mle"), ("res_rate_lo", "pi_lo"),
                    ("res_rate_hi", "pi_hi")):
        ses[c] = ses[pcol] * ses["n"] / INTERACTION_MIN
    ses["part_rate"] = (1 - ses["pi_mle"]) * ses["n"] / INTERACTION_MIN
    ses["part_rate_lo"] = (1 - ses["pi_hi"]) * ses["n"] / INTERACTION_MIN
    ses["part_rate_hi"] = (1 - ses["pi_lo"]) * ses["n"] / INTERACTION_MIN
    # sessions with zero dyad calls: rates are zero
    for c in ("res_rate", "part_rate", "res_rate_lo", "res_rate_hi",
              "part_rate_lo", "part_rate_hi"):
        ses.loc[ses["n"] == 0, c] = 0.0
    ses.to_csv(out / "v2_session_quantification.csv", index=False)

    wt = ses[ses["genotype"] == "WT"]
    het = ses[ses["genotype"] == "HET"]
    tests = []
    for metric, label in [
        ("alone_rate", "Alone-phase call rate (calls/min)"),
        ("dyad_rate", "Dyad call rate, interaction window (calls/min)"),
        ("res_rate", "RESIDENT-attributed rate, GEO pi-MLE (calls/min)"),
        ("part_rate", "PARTNER-attributed rate, GEO pi-MLE (calls/min)"),
        ("pi_mle", "Resident share of dyad calls (pi-MLE)"),
    ]:
        p, s = mw(wt[metric], het[metric])
        tests.append({"metric": metric, "label": label,
                      "wt_mean": float(wt[metric].mean()),
                      "het_mean": float(het[metric].mean()),
                      "wt_median": float(wt[metric].median()),
                      "het_median": float(het[metric].median()),
                      "mannwhitney_p": p, "sig": s})
    tstats = pd.DataFrame(tests)
    tstats.to_csv(out / "v2_attribution_statistics.csv", index=False)
    print(tstats[["label", "wt_mean", "het_mean", "mannwhitney_p", "sig"]]
          .to_string(index=False))

    # attributed 30-s time course from per-call posteriors
    tc = []
    att["bin"] = (att["t"] // 30).astype(int) * 30
    for (animal, b), g in att.groupby(["animal_id", "bin"]):
        tc.append({"animal_id": animal, "genotype": A.GENOTYPE_MAP[animal],
                   "bin_s": int(b), "res_calls": float(g["p_res_hmm"].sum()),
                   "part_calls": float((1 - g["p_res_hmm"]).sum())})
    al = calls[calls["phase"] == "alone"].copy()
    al["bin"] = (al["start_s"] // 30).astype(int) * 30
    for (animal, b), g in al.groupby(["animal_id", "bin"]):
        tc.append({"animal_id": str(animal),
                   "genotype": A.GENOTYPE_MAP[str(animal)],
                   "bin_s": int(b), "res_calls": float(len(g)),
                   "part_calls": 0.0})
    pd.DataFrame(tc).to_csv(out / "v2_timecourse.csv", index=False)

    big_het = het[het["n"] >= 3]
    summary = {
        "wt_res_rate_mean": float(wt["res_rate"].mean()),
        "het_res_rate_mean": float(het["res_rate"].mean()),
        "wt_part_rate_mean": float(wt["part_rate"].mean()),
        "het_part_rate_mean": float(het["part_rate"].mean()),
        "wt_pi_mean": float(wt["pi_mle"].mean()),
        "het_pi_mean": float(het["pi_mle"].mean()),
        "wt_pi_shuffle_mean": float(wt["pi_shuffle"].mean()),
        "het_pi_shuffle_mean": float(het["pi_shuffle"].mean()),
        "het_pi_mean_n3": float(big_het["pi_mle"].mean()) if len(big_het) else None,
        "het_total_dyad_calls": int(het["n"].sum()),
        "wt_total_dyad_calls": int(wt["n"].sum()),
    }
    with open(out / "v2_quant_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
