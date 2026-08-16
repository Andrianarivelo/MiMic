"""Figures for the call x behaviour analysis.

v4_fig1  WHERE calls happen   enrichment per behaviour, raw vs movement-adjusted
v4_fig2  WHEN calls happen    peri-call behaviour and peri-onset call rate
v4_fig3  WHAT FOLLOWS calls   matched-counterfactual consequences
v4_fig4  CALL TYPES           behavioural profile per syllable class / duration
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A

C_M1, C_M2 = "#1baf7a", "#4a3aa7"     # resident actor vs partner actor
C_WT, C_HET = "#2a78d6", "#eb6834"
C_POS, C_NEG = "#2a78d6", "#e34948"
C_INK, C_SEC, C_MUT = "#0b0b0b", "#52514e", "#898781"
C_GRID, C_SURF, C_BASE = "#e1e0d9", "#fcfcfb", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": C_SURF, "axes.facecolor": C_SURF,
    "savefig.facecolor": C_SURF, "axes.edgecolor": C_BASE,
    "axes.labelcolor": C_SEC, "text.color": C_INK,
    "xtick.color": C_MUT, "ytick.color": C_MUT,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "figure.titlesize": 14, "figure.titleweight": "bold", "svg.fonttype": "none",
})
OUT = A.ensure_out()

PRETTY = {
    "nose2anogenital": "anogenital sniff", "nose2body": "body sniff",
    "nose2nose": "nose-to-nose", "sidebyside": "side-by-side",
    "sidereside": "side-to-side", "oriented_toward": "oriented toward",
    "following": "following", "chasing": "chasing", "approach": "approach",
    "withdrawal_from_partner": "withdrawal", "escape": "escape",
    "withdrawal_after_contact": "withdrawal after contact",
    "fighting": "fighting", "rearing": "rearing", "passive": "passive",
    "ACTIVEsocial": "ACTIVE social engagement (composite)",
    "CONTACTmutual": "mutual contact (composite)",
    "speed_m1": "resident speed", "speed_m2": "partner speed",
    "dist": "inter-animal distance", "app_m1": "resident approach velocity",
    "app_m2": "partner approach velocity",
}


def pretty(flag):
    if flag[:2] in ("m1", "m2"):
        who = "resident" if flag[:2] == "m1" else "partner"
        return f"{PRETTY.get(flag[3:], flag[3:])} ({who})"
    return PRETTY.get(flag, flag)


def savefig(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load():
    d = {}
    d["enr"] = pd.read_csv(OUT / "beh_enrichment.csv")
    d["peri"] = pd.read_csv(OUT / "beh_peri_call.csv")
    d["onset"] = pd.read_csv(OUT / "beh_peri_onset.csv")
    d["cons"] = pd.read_csv(OUT / "beh_consequence.csv")
    d["cstat"] = pd.read_csv(OUT / "beh_consequence_stats.csv")
    d["prof"] = pd.read_csv(OUT / "beh_calltype.csv")
    d["pred"] = json.load(open(OUT / "beh_prediction.json"))
    d["val"] = json.load(open(OUT / "beh_validation.json"))
    d["enr"] = pd.read_csv(OUT / "beh_enrichment.csv")
    return d


# =====================================================================
def fig1(d):
    enr = d["enr"].copy()
    fig = plt.figure(figsize=(13.5, 8.8))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.42,
                          width_ratios=[1.35, 1.0, 1.0])

    # (a) enrichment forest, resident-actor flags
    ax = fig.add_subplot(gs[:, 0])
    sub = enr[enr["actor"] == "m1"].sort_values("log2_RR_adj")
    y = np.arange(len(sub))
    sig = sub["q_adj"] < 0.05
    ax.barh(y, sub["log2_RR_crude"], height=0.36, color=C_MUT, alpha=0.55,
            label="raw")
    ax.barh(y + 0.38, sub["log2_RR_adj"], height=0.36,
            color=[C_POS if v > 0 else C_NEG for v in sub["log2_RR_adj"]],
            label="adjusted for speed × distance")
    ax.axvline(0, color=C_INK, lw=1)
    for i, (_, r) in enumerate(sub.iterrows()):
        star = "*" if r["q_adj"] < 0.05 else ""
        ax.text(r["log2_RR_adj"] + (0.06 if r["log2_RR_adj"] > 0 else -0.06),
                i + 0.38, f"{star}", va="center", fontsize=11,
                ha="left" if r["log2_RR_adj"] > 0 else "right", color=C_INK)
    ax.set_yticks(y + 0.19,
                  [f"{PRETTY.get(b, b)}  ({o:.1f}%, n={int(c)})"
                   for b, o, c in zip(sub["behavior"], sub["occupancy_pct"],
                                      sub["calls_in_state"])], fontsize=8)
    ax.set_xlabel("log₂ call-rate ratio  (in state vs out)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_MUT, alpha=0.55, label="raw"),
                       Patch(color=C_POS, label="adjusted — more calls"),
                       Patch(color=C_NEG, label="adjusted — fewer calls")],
              fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title("(a)  Where calls happen — resident's own behaviour\n"
                 "* = q < 0.05 vs circular-shift null", loc="left")

    # (b) partner-actor flags
    ax = fig.add_subplot(gs[0, 1])
    sub2 = enr[enr["actor"] == "m2"].sort_values("log2_RR_adj")
    y2 = np.arange(len(sub2))
    ax.barh(y2, sub2["log2_RR_adj"],
            color=[C_POS if v > 0 else C_NEG for v in sub2["log2_RR_adj"]])
    for i, (_, r) in enumerate(sub2.iterrows()):
        if r["q_adj"] < 0.05:
            ax.text(r["log2_RR_adj"] * 1.02, i, "*", va="center", fontsize=11,
                    ha="left" if r["log2_RR_adj"] > 0 else "right")
    ax.axvline(0, color=C_INK, lw=1)
    ax.set_yticks(y2, [PRETTY.get(b, b) for b in sub2["behavior"]], fontsize=7.5)
    ax.set_xlabel("log₂ rate ratio (adjusted)")
    ax.set_title("(b)  … and the partner's behaviour", loc="left")

    # (c) actor asymmetry for directional behaviours
    ax = fig.add_subplot(gs[0, 2])
    asym = d["val"].get("actor_asymmetry", {})
    dirs = list(asym.get("per_behavior", {}).keys()) or [
        "nose2anogenital", "following", "chasing", "approach"]
    piv = enr.set_index(["behavior", "actor"])["log2_RR_adj"]
    xs = np.arange(len(dirs))
    v1 = [piv.get((b, "m1"), np.nan) for b in dirs]
    v2 = [piv.get((b, "m2"), np.nan) for b in dirs]
    ax.bar(xs - 0.2, v1, width=0.38, color=C_M1, label="resident is the actor")
    ax.bar(xs + 0.2, v2, width=0.38, color=C_M2, label="partner is the actor")
    ax.axhline(0, color=C_INK, lw=1)
    ax.set_xticks(xs, [PRETTY.get(b, b) for b in dirs], rotation=35,
                  ha="right", fontsize=7.5)
    ax.set_ylabel("log₂ rate ratio (adjusted)")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title(f"(c)  Enrichment follows the actor more than\nthe target: gap "
                 f"{asym.get('stat_log2', float('nan')):+.2f} log₂ "
                 f"({2**asym.get('stat_log2', 0):.1f}×), p = "
                 f"{asym.get('p', float('nan')):.2f} — suggestive only",
                 loc="left")

    # (d) predictive models
    ax = fig.add_subplot(gs[1, 1])
    p = d["pred"]
    tags = ["kinematics", "behaviour", "both"]
    means = [p[t]["loso_auc_mean"] for t in tags]
    sds = [p[t]["loso_auc_sd"] for t in tags]
    ax.bar(np.arange(3), means, yerr=sds, width=0.6,
           color=[C_MUT, C_M1, C_INK], capsize=3, edgecolor=C_SURF)
    for i, m in enumerate(means):
        ax.text(i, m + 0.012, f"{m:.3f}", ha="center", fontsize=8.5, color=C_SEC)
    ax.axhline(0.5, color=C_INK, lw=1.2, ls="--")
    ax.set_xticks(range(3), ["kinematics\nonly", "behaviour\nonly", "both"])
    ax.set_ylim(0.45, max(means) + 0.09)
    ax.set_ylabel("held-out AUC (leave-one-session-out)")
    ax.set_title("(d)  Predicting which 100 ms bins\ncontain a call", loc="left")

    # (e) split-half reliability
    ax = fig.add_subplot(gs[1, 2])
    v = d["val"]
    ax.barh([0], [v["splithalf_rho_mean"]],
            xerr=[[v["splithalf_rho_mean"] - v["splithalf_rho_lo"]],
                  [v["splithalf_rho_hi"] - v["splithalf_rho_mean"]]],
            color=C_M1, height=0.45, capsize=4)
    ax.axvline(0, color=C_INK, lw=1)
    ax.set_xlim(-0.2, 1.05)
    ax.set_yticks([0], ["behaviour\nprofile"])
    ax.set_xlabel("Spearman ρ between random halves of the cohort")
    ax.set_title(f"(e)  The profile replicates across\nanimals (ρ = "
                 f"{v['splithalf_rho_mean']:.2f})", loc="left")

    fig.suptitle("Where do calls happen? Behavioural context of "
                 f"{d['val']['n_calls_partner']:,} interaction-window calls", y=1.0)
    savefig(fig, "v4_fig1_where")


# =====================================================================
def fig2(d):
    peri, onset = d["peri"], d["onset"]
    enr = d["enr"]
    top = (enr[(enr["actor"] == "m1") & (enr["calls_in_state"] >= 100)]
           .sort_values("log2_RR_adj", ascending=False)["flag"].tolist())
    show = (top[:3] + [f for f in ["m1_withdrawal_from_partner", "m1_nose2nose"]
                       if f in set(enr["flag"])])[:5]
    fig, axes = plt.subplots(2, len(show), figsize=(3.0 * len(show), 6.4),
                             sharex=True)
    if len(show) == 1:
        axes = axes.reshape(2, 1)
    for c, flag in enumerate(show):
        s = peri[peri["flag"] == flag].sort_values("lag_s")
        ax = axes[0, c]
        ax.fill_between(s["lag_s"], s["null_lo"], s["null_hi"], color=C_MUT,
                        alpha=0.25, edgecolor="none")
        ax.plot(s["lag_s"], s["p_behavior"], color=C_M1, lw=2)
        ax.axvline(0, color=C_INK, lw=1, ls="--")
        ax.set_title(pretty(flag), fontsize=9)
        if c == 0:
            ax.set_ylabel("P(behaviour)\naround a call")
        o = onset[onset["flag"] == flag].sort_values("lag_s")
        ax2 = axes[1, c]
        if len(o):
            ax2.fill_between(o["lag_s"], o["null_lo"], o["null_hi"],
                             color=C_MUT, alpha=0.25, edgecolor="none")
            ax2.plot(o["lag_s"], o["call_rate"], color=C_M2, lw=2)
        ax2.axvline(0, color=C_INK, lw=1, ls="--")
        ax2.set_xlabel("time from event (s)")
        if c == 0:
            ax2.set_ylabel("call rate (/min)\naround behaviour onset")
    axes[0, 0].text(0.02, 0.96, "shaded = shift null (95%)", fontsize=7.5,
                    color=C_SEC, transform=axes[0, 0].transAxes, va="top")
    fig.suptitle("When do calls happen? Behaviour around calls (top) and calls "
                 "around behaviour onsets (bottom)", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, "v4_fig2_when")


# =====================================================================
def fig3(d):
    cst, cons = d["cstat"], d["cons"]
    EARLY = "early_0.2-1s"
    KIN = ["speed_m1", "speed_m2", "app_m1", "app_m2", "dist"]
    fig = plt.figure(figsize=(15.5, 5.4))
    gs = fig.add_gridspec(1, 4, wspace=0.55, width_ratios=[1.0, 1.15, 0.9, 1.0])

    sub = cst[(cst["role"] == "res") & (cst["window"] == EARLY)].copy()
    kin = sub[sub["metric"].isin(KIN)]

    def forest(ax, tab, unit, title):
        tab = tab.sort_values("pooled_diff")
        for i, (_, r) in enumerate(tab.iterrows()):
            ax.plot([r["boot_lo"], r["boot_hi"]], [i, i], color=C_MUT, lw=1.7)
            ax.scatter([r["pooled_diff"]], [i], s=36, zorder=3,
                       c=C_POS if r["pooled_diff"] > 0 else C_NEG,
                       edgecolors="white", linewidths=0.5)
            if r["q_boot"] < 0.05:
                ax.text(r["boot_hi"] if r["pooled_diff"] > 0 else r["boot_lo"],
                        i, "  *" if r["pooled_diff"] > 0 else "*  ",
                        va="center", fontsize=13, color=C_INK,
                        ha="left" if r["pooled_diff"] > 0 else "right")
        ax.axvline(0, color=C_INK, lw=1)
        ax.set_yticks(range(len(tab)), [pretty(m) for m in tab["metric"]],
                      fontsize=8)
        ax.set_xlabel(unit)
        ax.set_title(title, loc="left")
        ax.margins(x=0.18)

    forest(fig.add_subplot(gs[0, 0]), kin, "call − matched control (px/s; px for distance)",
           "(a)  Movement after a call, 0.2–1 s\n* = 95% CI excludes zero")
    forest(fig.add_subplot(gs[0, 1]), sub[~sub["metric"].isin(KIN)],
           "Δ probability, call − matched control",
           "(b)  Behaviour after a call, 0.2–1 s")

    # (c) per-session dots for the two strongest kinematic effects
    ax = fig.add_subplot(gs[0, 2])
    best = (kin.reindex(kin["p_boot"].sort_values().index)
            .head(2)["metric"].tolist())
    rng = np.random.default_rng(0)
    for i, m in enumerate(best):
        per = (cons[(cons["metric"] == m) & (cons["role"] == "res")
                    & (cons["window"] == EARLY)]
               .groupby("animal_id")["diff"].mean())
        ax.scatter(np.full(len(per), i) + rng.normal(0, 0.06, len(per)), per,
                   s=24, c=C_M1, alpha=0.75, edgecolors="white", linewidths=0.4)
        ax.hlines(per.median(), i - 0.24, i + 0.24, color=C_M1, lw=3)
    ax.axhline(0, color=C_INK, lw=1, ls="--")
    allv = np.concatenate([
        cons[(cons["metric"] == m) & (cons["role"] == "res")
             & (cons["window"] == EARLY)].groupby("animal_id")["diff"]
        .mean().to_numpy() for m in best])
    lim = np.nanpercentile(np.abs(allv), 92) * 1.25
    ax.set_ylim(-lim, lim)
    ax.set_xticks(range(len(best)),
                  [pretty(m).replace(" ", "\n", 1) for m in best], fontsize=8)
    ax.set_ylabel("per-session change (px/s)")
    ax.set_title("(c)  Consistent across sessions", loc="left")

    # (d) how it unfolds over time
    ax = fig.add_subplot(gs[0, 3])
    order = ["early_0.2-1s", "mid_0.2-2s", "late_2-5s"]
    show = [("app_m2", C_M2, "partner approach velocity"),
            ("speed_m1", C_M1, "caller speed"),
            ("dist", C_INK, "inter-animal distance")]
    xs = np.arange(len(order))
    for m, col, lab in show:
        vals, los, his = [], [], []
        for w in order:
            r = cst[(cst["role"] == "res") & (cst["window"] == w)
                    & (cst["metric"] == m)]
            if not len(r):
                vals.append(np.nan); los.append(np.nan); his.append(np.nan)
                continue
            r = r.iloc[0]
            # effect in units of its own CI half-width: |value| > 1 is exactly
            # "the 95% CI excludes zero"
            h = max((r["boot_hi"] - r["boot_lo"]) / 2, 1e-9)
            vals.append(r["pooled_diff"] / h)
            los.append(r["boot_lo"] / h)
            his.append(r["boot_hi"] / h)
        ax.plot(xs, vals, marker="o", ms=5, lw=1.8, color=col, label=lab)
    ax.axhline(0, color=C_INK, lw=1, ls="--")
    for yv in (-1, 1):
        ax.axhline(yv, color=C_BASE, lw=1, ls=":")
    ax.text(2.02, 1.05, "CI excludes 0", fontsize=7, color=C_MUT, ha="right")
    ax.set_xticks(xs, ["0.2–1 s", "0.2–2 s", "2–5 s"])
    ax.set_ylabel("effect / CI half-width")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.set_title("(d)  The sequence after a call", loc="left")

    fig.suptitle("What follows a call? Matched-counterfactual contrast "
                 "(resident-attributed calls)", y=1.03)
    savefig(fig, "v4_fig3_consequences")


# =====================================================================
def fig4(d):
    prof = d["prof"]
    val = d["val"]
    schemes = [s for s in ("syllable", "duration") if (prof["scheme"] == s).any()]
    fig, axes = plt.subplots(1, len(schemes) + 1,
                             figsize=(5.6 * len(schemes) + 4.4, 5.6),
                             gridspec_kw={"width_ratios": [1] * len(schemes) + [0.8]})
    axes = np.atleast_1d(axes)
    for k, scheme in enumerate(schemes):
        sub = prof[(prof["scheme"] == scheme) & (prof["flag"].str.startswith("m1"))]
        piv = sub.pivot_table(index="flag", columns="type", values="log2_ratio")
        keep = [f for f in piv.index
                if prof[(prof["scheme"] == scheme) & (prof["flag"] == f)]
                ["p_all_calls"].iloc[0] > 0.02]
        piv = piv.loc[keep]
        ax = axes[k]
        vmax = np.nanmax(np.abs(piv.to_numpy()))
        im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=35,
                      ha="right", fontsize=8)
        ax.set_yticks(range(piv.shape[0]),
                      [PRETTY.get(f[3:], f[3:]) for f in piv.index], fontsize=8)
        ax.grid(False)
        h = val["calltype_heterogeneity"].get(scheme, {})
        ax.set_title(f"({'abc'[k]})  {scheme} types\n"
                     f"heterogeneity p = {h.get('p', float('nan')):.3f} "
                     f"({h.get('n_types', 0)} types, n={h.get('n_calls', 0)})",
                     loc="left")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03,
                     label="log₂ (state share for this type / for all calls)")

    ax = axes[-1]
    ax.set_axis_off()
    p = d["pred"]
    coef = p["coefficients_both"]
    # drop coefficients estimated from states that almost never host a call:
    # they are unstable and would otherwise top the list on noise alone
    enr = d["enr"].set_index("flag")
    ok = {k for k in coef
          if k not in enr.index or enr.loc[k, "calls_in_state"] >= 30}
    top = sorted(((k, v) for k, v in coef.items() if k in ok),
                 key=lambda kv: -abs(kv[1]))[:11]
    txt = ("STRONGEST CORRELATES OF CALLING\n(multivariate, both cue sets;\n"
           "states with <30 calls dropped)\n\n")
    for k_, v_ in top:
        txt += f"  {pretty(k_)[:34]:34s} {v_:+.2f}\n"
    txt += (f"\nheld-out AUC\n"
            f"  kinematics only  {p['kinematics']['loso_auc_mean']:.3f}\n"
            f"  behaviour only   {p['behaviour']['loso_auc_mean']:.3f}\n"
            f"  both             {p['both']['loso_auc_mean']:.3f}\n")
    ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=8.2,
            family="monospace", color=C_INK,
            bbox=dict(boxstyle="round,pad=0.6", fc="#f9f9f7", ec=C_BASE))
    ax.set_title("(c)  What correlates best", loc="left")

    fig.suptitle("Do different call types occur in different behaviours?", y=1.02)
    fig.tight_layout()
    savefig(fig, "v4_fig4_calltypes")


def main():
    d = load()
    fig1(d)
    fig2(d)
    fig3(d)
    fig4(d)


if __name__ == "__main__":
    main()
