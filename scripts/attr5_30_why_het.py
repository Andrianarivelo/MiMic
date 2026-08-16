"""Why HET mice do not call: decomposing an 11.9x rate difference.

The session call rate factorises exactly as

    R = sum_s T_s * lambda_s

over mutually exclusive behavioural states s, with T_s the fraction of time in
s (OPPORTUNITY) and lambda_s the call rate while in s (PROPENSITY).  The
Kitagawa identity splits the genotype gap into the two without residual:

    R_W - R_H = sum_s (T_s^W - T_s^H) (lambda_s^W + lambda_s^H)/2      [opportunity]
              + sum_s (T_s^W + T_s^H)/2 (lambda_s^W - lambda_s^H)      [propensity]

A second factorisation separates starting from sustaining:

    R = (bouts per minute) x (calls per bout)                [initiation x maintenance]

and a third asks whether the deficit is a uniform gain change or a loss of
contextual modulation, by comparing the shape of lambda as a function of
inter-animal distance and of behavioural state across genotypes.
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

BIN_MIN = C.BIN_S / 60.0        # minutes per bin
RNG = np.random.default_rng(11)

GROUPS = {
    "investigate": ["nose2anogenital", "nose2body", "following", "chasing",
                    "approach", "oriented_toward"],
    "contact": ["nose2nose", "sidebyside", "sidereside", "fighting"],
    "withdraw": ["withdrawal_from_partner", "escape",
                 "withdrawal_after_contact"],
    "solitary": ["rearing", "passive"],
}


def state_of(bins: pd.DataFrame) -> pd.Series:
    """Coarse mutually exclusive state from the resident's flags."""
    s = pd.Series("none", index=bins.index, dtype=object)
    for g in ("solitary", "withdraw", "contact", "investigate"):
        m = np.zeros(len(bins), bool)
        for f in GROUPS[g]:
            m |= bins[f"m1_{f}"].to_numpy() > 0.5
        s[m] = g
    return s


# ---------------------------------------------------------------- Kitagawa
def kitagawa(bins: pd.DataFrame, n_boot=4000):
    bins = bins.copy()
    bins["state"] = state_of(bins)
    states = sorted(bins["state"].unique())
    sids = {g: sorted(bins.loc[bins.genotype == g, "animal_id"].unique())
            for g in ("WT", "HET")}

    def compute(sel_w, sel_h):
        res = {}
        for g, sel in (("WT", sel_w), ("HET", sel_h)):
            d = bins[bins["animal_id"].isin(sel)] if not isinstance(sel, pd.DataFrame) else sel
            tot = len(d)
            T = np.array([(d["state"] == s).sum() / tot for s in states])
            lam = np.array([
                d.loc[d["state"] == s, "n_calls"].sum()
                / max((d["state"] == s).sum() * BIN_MIN, 1e-9) for s in states])
            res[g] = (T, lam, d["n_calls"].sum() / (tot * BIN_MIN))
        (Tw, lw, Rw), (Th, lh, Rh) = res["WT"], res["HET"]
        opp = float(np.sum((Tw - Th) * (lw + lh) / 2))
        prop = float(np.sum((Tw + Th) / 2 * (lw - lh)))
        return {"R_WT": float(Rw), "R_HET": float(Rh), "gap": float(Rw - Rh),
                "opportunity": opp, "propensity": prop,
                "cf_HET_with_WT_time": float(np.sum(Tw * lh)),
                "cf_WT_with_HET_time": float(np.sum(Th * lw)),
                "T_WT": Tw.tolist(), "T_HET": Th.tolist(),
                "lam_WT": lw.tolist(), "lam_HET": lh.tolist()}

    obs = compute(sids["WT"], sids["HET"])
    obs["states"] = states
    draws = {k: [] for k in ("opportunity", "propensity", "gap",
                             "cf_HET_with_WT_time")}
    for _ in range(n_boot):
        w = list(RNG.choice(sids["WT"], len(sids["WT"]), replace=True))
        h = list(RNG.choice(sids["HET"], len(sids["HET"]), replace=True))
        try:
            r = compute(w, h)
        except Exception:                       # noqa: BLE001
            continue
        for k in draws:
            draws[k].append(r[k])
    for k, v in draws.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        obs[k + "_ci"] = [float(lo), float(hi)]
    obs["pct_propensity"] = 100 * obs["propensity"] / obs["gap"]
    pf = np.array(draws["propensity"]) / np.array(draws["gap"])
    obs["pct_propensity_ci"] = [float(x) for x in np.percentile(100 * pf, [2.5, 97.5])]
    return obs


# ------------------------------------------------------------------- bouts
def bout_structure(calls: pd.DataFrame, gap_s: float):
    rows = []
    for (aid, gt), g in calls.groupby(["animal_id", "genotype"]):
        t = np.sort(g["start_s"].to_numpy())
        ici = np.diff(t)
        newb = np.concatenate([[True], ici > gap_s])
        nb = int(newb.sum())
        rows.append({"animal_id": aid, "genotype": gt, "n_calls": len(t),
                     "n_bouts": nb, "calls_per_bout": len(t) / nb,
                     "bouts_per_min": nb / 10.0, "rate": len(t) / 10.0,
                     "within_ici_ms": float(np.median(ici[ici <= gap_s]) * 1000)
                     if np.any(ici <= gap_s) else np.nan,
                     "max_bout": int(np.max(np.diff(
                         np.flatnonzero(np.concatenate([newb, [True]]))))) if nb else 0})
    return pd.DataFrame(rows)


def pick_gap(calls: pd.DataFrame) -> tuple[float, dict]:
    """Bout criterion from a two-component exponential mixture of the ICIs."""
    ic = []
    for _, g in calls.groupby("animal_id"):
        t = np.sort(g["start_s"].to_numpy())
        ic.append(np.diff(t))
    ic = np.concatenate([x for x in ic if x.size])
    ic = ic[(ic > 0) & (ic < 300)]
    w, l1, l2 = 0.5, 1 / 0.2, 1 / 20.0
    for _ in range(500):
        a = w * l1 * np.exp(-l1 * ic)
        b = (1 - w) * l2 * np.exp(-l2 * ic)
        r = a / (a + b + 1e-300)
        w = r.mean()
        l1 = r.sum() / max((r * ic).sum(), 1e-12)
        l2 = (1 - r).sum() / max(((1 - r) * ic).sum(), 1e-12)
    gap = float(np.log((w * l1) / ((1 - w) * l2)) / (l1 - l2))
    return gap, {"w_fast": float(w), "tau_fast_s": float(1 / l1),
                 "tau_slow_s": float(1 / l2), "gap_s": gap, "n_ici": int(ic.size)}


# ------------------------------------------------------------- gain vs distance
def distance_gain(bins: pd.DataFrame, nq=8):
    d = bins.dropna(subset=["dist"]).copy()
    edges = np.nanpercentile(d["dist"], np.linspace(0, 100, nq + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    d["q"] = pd.cut(d["dist"], edges, labels=False, duplicates="drop")
    rows = []
    for (gt, q), g in d.groupby(["genotype", "q"]):
        rows.append({"genotype": gt, "q": int(q),
                     "dist_med": float(g["dist"].median()),
                     "rate": g["n_calls"].sum() / max(len(g) * BIN_MIN, 1e-9),
                     "n_bins": len(g), "n_calls": float(g["n_calls"].sum())})
    gain = pd.DataFrame(rows)
    # parallel-slopes test on the log scale (Poisson GLM with offset)
    import statsmodels.api as sm
    dd = gain[gain["n_calls"] > 0].copy()
    dd["logd"] = np.log(dd["dist_med"])
    dd["het"] = (dd["genotype"] == "HET").astype(float)
    X = sm.add_constant(pd.DataFrame({
        "logd": dd["logd"], "het": dd["het"], "inter": dd["logd"] * dd["het"]}))
    m = sm.GLM(dd["n_calls"], X, family=sm.families.Poisson(),
               offset=np.log(dd["n_bins"] * BIN_MIN)).fit()
    return gain, {"beta_logd": float(m.params["logd"]),
                  "beta_het": float(m.params["het"]),
                  "beta_interaction": float(m.params["inter"]),
                  "p_interaction": float(m.pvalues["inter"]),
                  "p_het": float(m.pvalues["het"])}


# --------------------------------------------------------------- latency
def logrank(t1, e1, t2, e2):
    t = np.concatenate([t1, t2])
    e = np.concatenate([e1, e2])
    g = np.concatenate([np.zeros(len(t1)), np.ones(len(t2))])
    order = np.argsort(t)
    t, e, g = t[order], e[order], g[order]
    O, E, V = 0.0, 0.0, 0.0
    for ti in np.unique(t[e == 1]):
        at = t >= ti
        n, n1 = at.sum(), (at & (g == 1)).sum()
        d = int(((t == ti) & (e == 1)).sum())
        d1 = int(((t == ti) & (e == 1) & (g == 1)).sum())
        O += d1
        E += d * n1 / n
        if n > 1:
            V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    z = (O - E) / np.sqrt(V) if V > 0 else 0.0
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def main():
    out = A.ensure_out()
    bins = C.load_bins()
    calls = pd.read_csv(out / "v5_calls.csv", low_memory=False)
    res: dict = {}

    # ---- 1. Kitagawa -----------------------------------------------------
    print("=== 1. Kitagawa decomposition ===")
    k = kitagawa(bins)
    res["kitagawa"] = k
    print(f"  WT {k['R_WT']:.3f} vs HET {k['R_HET']:.3f} calls/min  "
          f"(gap {k['gap']:.3f}, {k['R_WT']/k['R_HET']:.1f}x)")
    print(f"  opportunity  {k['opportunity']:+.3f} "
          f"[{k['opportunity_ci'][0]:+.3f}, {k['opportunity_ci'][1]:+.3f}]")
    print(f"  propensity   {k['propensity']:+.3f} "
          f"[{k['propensity_ci'][0]:+.3f}, {k['propensity_ci'][1]:+.3f}]")
    print(f"  propensity share {k['pct_propensity']:.1f}% "
          f"[{k['pct_propensity_ci'][0]:.1f}, {k['pct_propensity_ci'][1]:.1f}]")
    print(f"  counterfactual: HET given WT's time budget -> "
          f"{k['cf_HET_with_WT_time']:.3f} calls/min")
    for s, tw, th, lw, lh in zip(k["states"], k["T_WT"], k["T_HET"],
                                 k["lam_WT"], k["lam_HET"]):
        print(f"    {s:>12}: time {100*tw:5.1f}% vs {100*th:5.1f}%   "
              f"rate {lw:6.2f} vs {lh:5.2f} /min  ({lw/max(lh,1e-9):5.1f}x)")

    # ---- 2. bouts --------------------------------------------------------
    print("\n=== 2. initiation vs maintenance ===")
    gap, mix = pick_gap(calls)
    print(f"  ICI mixture: {100*mix['w_fast']:.0f}% fast "
          f"(tau={1000*mix['tau_fast_s']:.0f} ms) / slow tau={mix['tau_slow_s']:.1f} s"
          f"  -> bout criterion {gap:.2f} s")
    bs = bout_structure(calls, gap)
    res["bout_mixture"] = mix
    for col in ("rate", "bouts_per_min", "calls_per_bout", "within_ici_ms",
                "max_bout"):
        w = bs.loc[bs.genotype == "WT", col].dropna()
        h = bs.loc[bs.genotype == "HET", col].dropna()
        u = stats.mannwhitneyu(w, h)
        print(f"  {col:>16}: WT {w.median():8.2f}  HET {h.median():7.2f}  "
              f"ratio {w.median()/max(h.median(),1e-9):5.2f}x  p={u.pvalue:.4f}")
        res.setdefault("bouts", {})[col] = {
            "WT_median": float(w.median()), "HET_median": float(h.median()),
            "p": float(u.pvalue)}
    bs.to_csv(out / "v5_bouts.csv", index=False)

    # ---- 3. escalation ---------------------------------------------------
    print("\n=== 3. escalation ===")
    tab = bs.groupby("genotype")["n_calls"]
    for thr in (30, 50, 100):
        w = int((bs[bs.genotype == "WT"].n_calls >= thr).sum())
        h = int((bs[bs.genotype == "HET"].n_calls >= thr).sum())
        p = stats.fisher_exact([[w, 12 - w], [h, 12 - h]])[1]
        print(f"  sessions with >= {thr:3d} calls: WT {w}/12  HET {h}/12  "
              f"Fisher p={p:.4f}")
        res.setdefault("escalation", {})[f"ge_{thr}"] = {
            "WT": w, "HET": h, "p": float(p)}
    print(f"  median calls/session: WT {tab.median()['WT']:.0f}  "
          f"HET {tab.median()['HET']:.0f}   max: WT {tab.max()['WT']:.0f}  "
          f"HET {tab.max()['HET']:.0f}")
    res["escalation"]["median_WT"] = float(tab.median()["WT"])
    res["escalation"]["median_HET"] = float(tab.median()["HET"])
    res["escalation"]["p_rate"] = float(stats.mannwhitneyu(
        bs[bs.genotype == "WT"].rate, bs[bs.genotype == "HET"].rate).pvalue)

    # ---- 4. repertoire recruitment --------------------------------------
    print("\n=== 4. repertoire recruits with rate (is HET acoustically odd?) ===")
    calls["ultra"] = calls["rf_dur_ms"] < 9
    per = calls.groupby(["animal_id", "genotype"]).agg(
        n=("call_id", "size"), fr_ultra=("ultra", "mean"),
        dur=("rf_dur_ms", "median")).reset_index()
    per["lograte"] = np.log10(per["n"] / 10.0)
    # SESSION-level ANCOVA (24 points).  Weighting by call count would treat
    # every call as independent and manufacture absurd p-values.
    import statsmodels.api as sm
    k = per["fr_ultra"] * per["n"]
    per["elogit"] = np.log((k + 0.5) / (per["n"] - k + 0.5))
    X = sm.add_constant(pd.DataFrame({
        "lograte": per["lograte"],
        "het": (per["genotype"] == "HET").astype(float)}))
    m = sm.OLS(per["elogit"], X).fit(cov_type="HC1")
    rho_w = stats.spearmanr(per[per.genotype == "WT"].n,
                            per[per.genotype == "WT"].fr_ultra)
    print(f"  within WT: rho(n_calls, frac ultrashort) = {rho_w.statistic:+.2f} "
          f"(p={rho_w.pvalue:.4f})")
    print("  session-level ANCOVA, empirical-logit(frac ultrashort) "
          "~ log10(rate) + genotype (n=24):")
    print(f"     log10(rate) {m.params['lograte']:+.2f} "
          f"(p={m.pvalues['lograte']:.2g})")
    print(f"     genotype    {m.params['het']:+.2f} (p={m.pvalues['het']:.3f})"
          f"  <- ns means HET sit ON the WT curve, not off it")
    lo = per[(per.genotype == "WT") & (per.n <= 40)]
    het_ = per[per.genotype == "HET"]
    pm = stats.mannwhitneyu(het_.fr_ultra, lo.fr_ultra)
    print(f"  rate-matched check: {len(lo)} WT sessions with <=40 calls "
          f"({lo.fr_ultra.median():.2f} ultrashort) vs 12 HET "
          f"({het_.fr_ultra.median():.2f}), MWU p = {pm.pvalue:.3f}")
    res["recruitment"] = {
        "rho_WT": float(rho_w.statistic), "p_rho_WT": float(rho_w.pvalue),
        "beta_lograte": float(m.params["lograte"]),
        "p_lograte": float(m.pvalues["lograte"]),
        "beta_genotype": float(m.params["het"]),
        "p_genotype": float(m.pvalues["het"]),
        "ratematched_WT_median": float(lo.fr_ultra.median()),
        "ratematched_HET_median": float(het_.fr_ultra.median()),
        "ratematched_p": float(pm.pvalue), "n_ratematched_WT": int(len(lo))}
    per.to_csv(out / "v5_recruitment.csv", index=False)

    # ---- 5. is the CONTEXT of calling preserved? -------------------------
    print("\n=== 5. do HET call in the same places? ===")
    bins2 = bins.copy()
    bins2["state"] = state_of(bins2)
    prof = {}
    for g in ("WT", "HET"):
        d = bins2[bins2.genotype == g]
        occ = d.groupby("state").size() / len(d)
        cal = d.groupby("state")["n_calls"].sum() / max(d["n_calls"].sum(), 1)
        prof[g] = (cal / occ).reindex(sorted(occ.index)).fillna(0)
    comp = pd.DataFrame(prof)
    rho = stats.spearmanr(comp["WT"], comp["HET"])
    print(comp.round(2).to_string())
    print(f"  Spearman rho(WT enrichment, HET enrichment) = {rho.statistic:+.2f} "
          f"(p={rho.pvalue:.3f}, {len(comp)} states)")
    pw = bins2[bins2.genotype == "WT"].groupby("state")["n_calls"].sum()
    ph = bins2[bins2.genotype == "HET"].groupby("state")["n_calls"].sum()
    tab2 = pd.concat([pw, ph], axis=1).fillna(0).to_numpy()
    chi = stats.chi2_contingency(tab2 + 0.5)
    print(f"  chi2 on P(state | call): p = {chi.pvalue:.3f}"
          f"  (large p = same contexts)")
    comp.to_csv(out / "v5_context_profile.csv")
    res["context"] = {"rho": float(rho.statistic), "p_rho": float(rho.pvalue),
                      "chi2_p": float(chi.pvalue),
                      "enrichment": comp.to_dict()}

    # ---- 5b. SOCIAL GAIN: how much does social context multiply calling? --
    print("\n=== 5b. social gain (call rate in social states / non-social) ===")

    def social_gain(sel_ids):
        d = bins2[bins2["animal_id"].isin(sel_ids)]
        soc = d["state"].isin(["investigate", "contact"])
        r_soc = d.loc[soc, "n_calls"].sum() / max(soc.sum() * BIN_MIN, 1e-9)
        r_non = d.loc[d["state"] == "none", "n_calls"].sum() / \
            max((d["state"] == "none").sum() * BIN_MIN, 1e-9)
        return r_soc, r_non, r_soc / max(r_non, 1e-9)

    ids = {g: sorted(bins2.loc[bins2.genotype == g, "animal_id"].unique())
           for g in ("WT", "HET")}
    gw, gh, ratios = [], [], []
    ow = social_gain(ids["WT"])
    oh = social_gain(ids["HET"])
    for _ in range(4000):
        gw.append(social_gain(list(RNG.choice(ids["WT"], 12, replace=True)))[2])
        gh.append(social_gain(list(RNG.choice(ids["HET"], 12, replace=True)))[2])
        ratios.append(gw[-1] / max(gh[-1], 1e-9))
    ciw = np.percentile(gw, [2.5, 97.5])
    cih = np.percentile(gh, [2.5, 97.5])
    cir = np.percentile(ratios, [2.5, 97.5])
    p_int = 2 * min(np.mean(np.array(ratios) <= 1), np.mean(np.array(ratios) >= 1))
    print(f"  WT : social {ow[0]:6.2f} vs non-social {ow[1]:5.2f} calls/min"
          f"  -> gain {ow[2]:.2f}x [{ciw[0]:.2f}, {ciw[1]:.2f}]")
    print(f"  HET: social {oh[0]:6.2f} vs non-social {oh[1]:5.2f} calls/min"
          f"  -> gain {oh[2]:.2f}x [{cih[0]:.2f}, {cih[1]:.2f}]")
    print(f"  gain ratio WT/HET = {ow[2]/oh[2]:.2f}x [{cir[0]:.2f}, {cir[1]:.2f}]"
          f"  bootstrap p = {p_int:.4f}")
    print("  (a genotype difference in GAIN, not just level, means the social "
          "modulation itself is lost)")
    res["social_gain"] = {
        "WT_social": ow[0], "WT_nonsocial": ow[1], "WT_gain": ow[2],
        "WT_gain_ci": ciw.tolist(),
        "HET_social": oh[0], "HET_nonsocial": oh[1], "HET_gain": oh[2],
        "HET_gain_ci": cih.tolist(),
        "gain_ratio": ow[2] / oh[2], "gain_ratio_ci": cir.tolist(),
        "p": float(p_int)}

    # ---- 6. gain vs distance --------------------------------------------
    print("\n=== 6. proximity gain function ===")
    gain, gstat = distance_gain(bins)
    gain.to_csv(out / "v5_distance_gain.csv", index=False)

    # session-level bootstrap for the slopes (the pooled GLM ignores clustering)
    def slope(sel_ids):
        d = bins[bins["animal_id"].isin(sel_ids)].dropna(subset=["dist"])
        e = np.nanpercentile(bins["dist"].dropna(), np.linspace(0, 100, 9))
        e[0], e[-1] = -np.inf, np.inf
        q = pd.cut(d["dist"], e, labels=False, duplicates="drop")
        r, x = [], []
        for qi, g in d.groupby(q):
            n = len(g)
            if n < 200:
                continue
            r.append(np.log((g["n_calls"].sum() + 0.5) / (n * BIN_MIN)))
            x.append(np.log(g["dist"].median()))
        if len(r) < 4:
            return np.nan
        return float(np.polyfit(x, r, 1)[0])

    sw, sh, sd = [], [], []
    obs_w, obs_h = slope(ids["WT"]), slope(ids["HET"])
    for _ in range(2000):
        a = slope(list(RNG.choice(ids["WT"], 12, replace=True)))
        b = slope(list(RNG.choice(ids["HET"], 12, replace=True)))
        sw.append(a)
        sh.append(b)
        sd.append(a - b)
    sd = np.array(sd)
    ci_d = np.nanpercentile(sd, [2.5, 97.5])
    p_d = 2 * min(np.nanmean(sd <= 0), np.nanmean(sd >= 0))
    print(f"  d log(rate) / d log(distance):  WT {obs_w:+.2f} "
          f"[{np.nanpercentile(sw,2.5):+.2f}, {np.nanpercentile(sw,97.5):+.2f}]"
          f"   HET {obs_h:+.2f} "
          f"[{np.nanpercentile(sh,2.5):+.2f}, {np.nanpercentile(sh,97.5):+.2f}]")
    print(f"  difference {obs_w-obs_h:+.2f} [{ci_d[0]:+.2f}, {ci_d[1]:+.2f}] "
          f"bootstrap p = {p_d:.4f}")
    print("  (WT call more the closer they are; a flat HET slope means "
          "proximity no longer drives calling)")
    gstat.update({"slope_WT": obs_w, "slope_HET": obs_h,
                  "slope_diff": obs_w - obs_h, "slope_diff_ci": ci_d.tolist(),
                  "slope_diff_p": float(p_d)})
    res["gain"] = gstat

    # ---- 7. latency ------------------------------------------------------
    print("\n=== 7. latency to first call after partner introduction ===")
    lat = []
    for (aid, gt), g in calls.groupby(["animal_id", "genotype"]):
        lat.append({"animal_id": aid, "genotype": gt,
                    "latency_s": float(g["start_s"].min() - C.T0), "event": 1})
    lat = pd.DataFrame(lat)
    z, p = logrank(lat[lat.genotype == "WT"].latency_s.to_numpy(),
                   lat[lat.genotype == "WT"].event.to_numpy(),
                   lat[lat.genotype == "HET"].latency_s.to_numpy(),
                   lat[lat.genotype == "HET"].event.to_numpy())
    print(f"  median latency: WT {lat[lat.genotype=='WT'].latency_s.median():.0f} s"
          f"   HET {lat[lat.genotype=='HET'].latency_s.median():.0f} s")
    print(f"  log-rank z = {z:+.2f}, p = {p:.3f}")
    lat.to_csv(out / "v5_latency.csv", index=False)
    res["latency"] = {"WT_median": float(lat[lat.genotype == "WT"].latency_s.median()),
                      "HET_median": float(lat[lat.genotype == "HET"].latency_s.median()),
                      "logrank_z": z, "p": p}

    # ---- 8. is the PARTNER different? -----------------------------------
    print("\n=== 8. control: do the partners behave differently? ===")
    pr = {}
    for f in C.FLAGS:
        w = bins[bins.genotype == "WT"].groupby("animal_id")[f"m2_{f}"].mean()
        h = bins[bins.genotype == "HET"].groupby("animal_id")[f"m2_{f}"].mean()
        pr[f] = {"WT": float(w.mean()), "HET": float(h.mean()),
                 "p": float(stats.mannwhitneyu(w, h).pvalue)}
    prd = pd.DataFrame(pr).T.sort_values("p")
    prd["q"] = C.bh(prd["p"].to_numpy())
    print(prd.head(6).to_string(float_format=lambda v: f"{v:.4f}"))
    res["partner_behaviour"] = prd.to_dict()
    prd.to_csv(out / "v5_partner_behaviour.csv")

    json.dump(res, open(out / "v5_why_het.json", "w"), indent=2, default=float)
    print("\nwrote", out / "v5_why_het.json")


if __name__ == "__main__":
    main()
