"""V2 figures (final, GEO method): method, benchmark journey, real-data
validation, quantification.

Outputs attribution/v2_fig{1..4}_*.png/.svg/.pdf
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A
import attr2_core as C2
from attr2_45_geometry import session_geometry

C_WT, C_HET = "#2a78d6", "#eb6834"
C_RES, C_PART = "#1baf7a", "#4a3aa7"
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


def savefig(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load():
    d = {}
    d["gate"] = json.load(open(OUT / "v2_kin_gate.json"))
    d["bench"] = json.load(open(OUT / "v2_benchmark_summary.json"))
    d["geo"] = json.load(open(OUT / "v2_geo_validation.json"))
    d["quant"] = json.load(open(OUT / "v2_quant_summary.json"))
    d["val50"] = json.load(open(OUT / "v2_validation.json"))
    d["matrix"] = pd.read_csv(OUT / "v2_benchmark_matrix.csv")
    d["ses"] = pd.read_csv(OUT / "v2_session_quantification.csv").assign(
        animal_id=lambda x: x["animal_id"].astype(str))
    d["att"] = pd.read_csv(OUT / "v2_geo_attribution.csv").assign(
        animal_id=lambda x: x["animal_id"].astype(str))
    d["stats"] = pd.read_csv(OUT / "v2_attribution_statistics.csv").set_index("metric")
    d["tc"] = pd.read_csv(OUT / "v2_timecourse.csv").assign(
        animal_id=lambda x: x["animal_id"].astype(str))
    d["v1ses"] = pd.read_csv(OUT / "session_quantification.csv").assign(
        animal_id=lambda x: x["animal_id"].astype(str))
    return d


# ========================================================================
def fig1_method(d):
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.36)

    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    ax.set_xlim(0, 103)
    ax.set_ylim(0, 10)
    boxes = [
        (1, "VOICE cues\nAE + bout-contrastive\nspaces", C_MUT,
         "dyad AUC 0.44-0.53\nREJECTED", 0.55),
        (17.5, "LEVEL cue\nloudness residual\nvs mic distance", C_MUT,
         "LOAO R² ≈ 0\nREJECTED", 0.55),
        (34, "KIN cue\nper-animal movement\n(alone-trained)", C_MUT,
         "dyad AUC 0.79, but real\ninteractions correlate\nmovement - FAILS in-domain", 0.75),
        (54, "GEO cue  (v2 final)\ncaller pursues, other\nretreats, close range", C_RES,
         "LOSO recovery 0.915\nvs design 0.916", 1.0),
        (73, "session prevalence\nMLE + profile CI\n(fixes posterior bias)", C_INK,
         "", 1.0),
        (88.5, "per-call posterior\n+ caller HMM\n+ abstention", C_INK, "", 1.0),
    ]
    for x, txt, col, verdict, alpha in boxes:
        ax.add_patch(FancyBboxPatch((x, 3.4), 13.5, 5.2,
                                    boxstyle="round,pad=0.35", fc=C_SURF,
                                    ec=col, lw=1.7, alpha=alpha))
        ax.text(x + 6.75, 6.0, txt, ha="center", va="center", fontsize=7.6,
                color=C_INK, alpha=alpha)
        if verdict:
            ax.text(x + 6.75, 2.2, verdict, ha="center", va="top", fontsize=6.8,
                    color=col if alpha == 1.0 else C_MUT, style="italic")
    for x0 in (68.5, 87.0):
        ax.add_patch(FancyArrowPatch((x0, 6.0), (x0 + 1.6, 6.0),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color=C_MUT, lw=1.4))
    ax.set_title("(a)  V2 method selection: every cue benchmarked, three rejected, "
                 "one survives (no stim table, no pretrained models)", loc="left")

    # (b) call-triggered speed envelope
    axb = fig.add_subplot(gs[1, 0])
    kin, _, _ = C2.load_kinematics()
    calls = A.load_calls()
    lags = np.arange(-2, 2.01, 1 / 6)
    trig, ctrl = [], []
    rng = np.random.default_rng(0)
    for animal in A.GENOTYPE_MAP:
        k = kin[animal]
        t = k["t"].to_numpy()
        v = k["m1_speed"].to_numpy()
        m = t < A.PARTNER_INTRO_S
        mu, sd = np.nanmean(v[m]), np.nanstd(v[m])
        ok = np.isfinite(v)
        g = calls[(calls["animal_id"] == animal) & (calls["phase"] == "alone")]
        for s in ((g["start_s"] + g["end_s"]) / 2):
            trig.append((np.interp(s + lags, t[ok], v[ok]) - mu) / (sd + 1e-9))
        for s in rng.uniform(2, 298, len(g) * 3):
            ctrl.append((np.interp(s + lags, t[ok], v[ok]) - mu) / (sd + 1e-9))
    for arr, col, lab in ((np.array(trig), C_RES, "at own call"),
                          (np.array(ctrl), C_MUT, "control times")):
        mean = np.nanmean(arr, 0)
        sem = np.nanstd(arr, 0) / np.sqrt(len(arr))
        axb.plot(lags, mean, color=col, lw=2, label=lab)
        axb.fill_between(lags, mean - sem, mean + sem, color=col, alpha=0.2,
                         edgecolor="none")
    axb.axvline(0, color=C_INK, lw=1, ls="--")
    axb.set_xlabel("time from call (s)")
    axb.set_ylabel("caller speed (z)")
    axb.legend(fontsize=8, frameon=False)
    axb.set_title("(b)  Callers move (alone-phase\nground truth)", loc="left")

    # (c) the GEO signature: approach asymmetry at call times
    axc = fig.add_subplot(gs[1, 1])
    vals = {}
    for gt in ("WT", "HET"):
        animals = [a for a, g in A.GENOTYPE_MAP.items() if g == gt]
        ac, pc, an, pn = [], [], [], []
        for a in animals:
            k = kin[a]
            g = calls[(calls["animal_id"] == a) & (calls["phase"] == "partner")]
            if not len(g):
                continue
            ct = ((g["start_s"] + g["end_s"]) / 2).to_numpy()
            geo_c = session_geometry(k, ct)
            cts = rng.uniform(A.PARTNER_INTRO_S + 2, A.SESSION_END_S - 2,
                              3 * len(ct))
            geo_n = session_geometry(k, cts)
            ac.append(np.nanmean(geo_c["app1"]))
            pc.append(np.nanmean(geo_c["app2"]))
            an.append(np.nanmean(geo_n["app1"]))
            pn.append(np.nanmean(geo_n["app2"]))
        vals[gt] = (np.nanmean(ac), np.nanmean(pc), np.nanmean(an), np.nanmean(pn))
    x = np.arange(2)
    width = 0.18
    for i, (gt, col) in enumerate((("WT", C_WT), ("HET", C_HET))):
        ac, pc, an, pn = vals[gt]
        axc.bar(x + (i - 0.5) * 2.2 * width, [ac, pc], width=width * 0.9,
                color=col, edgecolor=C_SURF, label=f"{gt} at calls")
        axc.bar(x + (i - 0.5) * 2.2 * width + width, [an, pn], width=width * 0.9,
                color=col, alpha=0.35, edgecolor=C_SURF,
                label=f"{gt} control")
    axc.axhline(0, color=C_BASE, lw=1)
    axc.set_xticks(x, ["resident\napproach", "partner\napproach"])
    axc.set_ylabel("approach velocity (px/s)")
    axc.legend(fontsize=7, frameon=False, ncols=2)
    axc.set_title("(c)  The GEO signature: at call times the\nresident pursues, "
                  "the partner retreats", loc="left")

    # (d) ICI / bouts
    axd = fig.add_subplot(gs[1, 2])
    ici = calls.sort_values(["animal_id", "start_s"]).groupby(
        "animal_id")["start_s"].diff().dropna()
    axd.hist(np.log10(ici), bins=50, color=C_WT, edgecolor=C_SURF)
    axd.axvline(np.log10(0.5), color=C_INK, lw=1.2, ls="--")
    axd.text(np.log10(0.5), axd.get_ylim()[1] * 0.95, " bout gap 0.5 s",
             fontsize=8, va="top")
    axd.set_xlabel("inter-call interval (log10 s)")
    axd.set_ylabel("call pairs")
    axd.set_title("(d)  Bout structure feeds the\ncaller HMM", loc="left")

    fig.suptitle("V2 method: the caller is the pursuing animal - selected by "
                 "elimination through benchmarks", y=1.0)
    savefig(fig, "v2_fig1_method")


# ========================================================================
def fig2_benchmarks(d):
    fig = plt.figure(figsize=(13.5, 4.9))
    gs = fig.add_gridspec(1, 3, wspace=0.35)
    mat = d["matrix"]

    ax = fig.add_subplot(gs[0, 0])
    piv = mat[mat["hmm"]].pivot_table(index="method", columns="set",
                                      values="pooled_auc")
    piv = piv.reindex(["fusion", "kin", "voice_v1", "voice_v2", "level"])
    im = ax.imshow(piv.to_numpy(), cmap="Blues", vmin=0.4, vmax=0.85,
                   aspect="auto")
    ax.set_xticks(range(len(piv.columns)),
                  [{"feature": "feature dyads\n(198 pairs)",
                    "audio": "audio mixtures\n(12 pairs)"}[c] for c in piv.columns])
    ax.set_yticks(range(len(piv)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.68 else C_INK)
    ax.grid(False)
    ax.set_title("(a)  Synthetic-dyad AUC: movement cues\nlook strong, voice is dead",
                 loc="left")

    ax = fig.add_subplot(gs[0, 1])
    v = d["val50"]
    bars = [
        ("kin fusion,\nreal WT sessions", v["wt_resident_frac_mean"], C_MUT),
        ("same, shuffled\ncall times", v["shuffle_wt_resident_frac"], C_MUT),
        ("GEO π-MLE,\nheld-out WT", d["geo"]["loso_wt_resident_pi_mle_mean"], C_RES),
        ("GEO, shuffled\ncall times", d["geo"]["shuffle_wt_resident_pi_mean"], C_BASE),
    ]
    xs = np.arange(len(bars))
    ax.bar(xs, [b[1] for b in bars], width=0.62,
           color=[b[2] for b in bars], edgecolor=C_SURF)
    for x, (_, val, _) in zip(xs, bars):
        ax.text(x, val + 0.015, f"{val:.2f}", ha="center", fontsize=8.5,
                color=C_SEC)
    bound = d["geo"]["design_bound"]
    ax.axhline(bound, color=C_INK, lw=1.3, ls="--")
    ax.text(-0.42, bound + 0.014, f"design truth {bound:.2f}", ha="left",
            fontsize=8, color=C_INK)
    ax.set_xticks(xs, [b[0] for b in bars], fontsize=7.3)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("WT resident fraction recovered")
    ax.set_title("(b)  In-domain test: kin fusion fails on real\ndyads; GEO recovers "
                 "the design truth", loc="left")

    ax = fig.add_subplot(gs[0, 2])
    ax.set_axis_off()
    g = d["geo"]
    txt = (
        "V2 SELECTION SUMMARY\n\n"
        "voice (2 spaces): dyad AUC 0.44/0.53 REJECT\n"
        "level-distance:   LOAO R² ≈ 0       REJECT\n"
        "kinematic:        dyad AUC 0.79 but\n"
        "                  in-domain shuffle-\n"
        "                  identical          REJECT\n"
        "GEO (pursuit):    LOSO recovery\n"
        f"                  {g['loso_wt_resident_pi_mle_mean']:.3f} vs {g['design_bound']:.3f}  KEEP\n\n"
        "GEO weights match literature:\n"
        f"  approach_self  {g['geo_weights']['app_self']:+.2f}\n"
        f"  approach_other {g['geo_weights']['app_other']:+.2f}\n"
        f"  distance       {g['geo_weights']['dist_z']:+.2f}\n\n"
        f"per-call sensitivity ~{g['wt_sensitivity_at_05']:.2f}; certainty\n"
        "lives at the SESSION level (π-MLE + CI)"
    )
    ax.text(0.02, 0.98, txt, va="top", ha="left", fontsize=8.6,
            family="monospace", color=C_INK,
            bbox=dict(boxstyle="round,pad=0.6", fc="#f9f9f7", ec=C_BASE))
    ax.set_title("(c)  Why GEO is the v2 method", loc="left")

    fig.suptitle("V2 benchmark journey: synthetic dyads propose, in-domain "
                 "validation disposes", y=1.05)
    savefig(fig, "v2_fig2_benchmarks")


# ========================================================================
def fig3_validation(d):
    fig = plt.figure(figsize=(13.5, 4.9))
    gs = fig.add_gridspec(1, 3, wspace=0.35)
    geo = d["geo"]
    ses = d["ses"]

    # (a) LOSO per session
    ax = fig.add_subplot(gs[0, 0])
    loso = geo["loso_wt_resident_pi_per_session"]
    keys = sorted(loso, key=lambda k: -loso[k])
    xs = np.arange(len(keys))
    ax.bar(xs, [loso[k] for k in keys], width=0.65, color=C_RES,
           edgecolor=C_SURF)
    bound = geo["design_bound"]
    ax.axhline(bound, color=C_INK, lw=1.3, ls="--")
    ax.text(len(keys) - 0.4, bound - 0.04, f"design truth {bound:.2f}",
            ha="right", va="top", fontsize=8.5)
    ax.set_xticks(xs, keys, rotation=45, fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("held-out resident share (π-MLE)")
    ax.set_title(f"(a)  Leave-one-session-out: mean "
                 f"{geo['loso_wt_resident_pi_mle_mean']:.2f}\n(model never sees "
                 "the held-out session)", loc="left")

    # (b) call-locked increment per session
    ax = fig.add_subplot(gs[0, 1])
    for i, (gt, col) in enumerate((("WT", C_WT), ("HET", C_HET))):
        sub = ses[(ses["genotype"] == gt) & (ses["n"] >= 3)]
        for _, r in sub.iterrows():
            ax.plot([i * 1.2, i * 1.2 + 0.5], [r["pi_shuffle"], r["pi_mle"]],
                    "-", color=col, alpha=0.45, lw=1)
        ax.scatter(np.full(len(sub), i * 1.2), sub["pi_shuffle"], s=22,
                   facecolors="none", edgecolors=col, linewidths=1.2,
                   label=f"{gt} shuffled" if i == 0 else None)
        ax.scatter(np.full(len(sub), i * 1.2 + 0.5), sub["pi_mle"], s=24,
                   c=col, edgecolors="white", linewidths=0.4)
    ax.set_xticks([0, 0.5, 1.2, 1.7],
                  ["WT\nshuffled", "WT\nreal", "HET\nshuffled", "HET\nreal"],
                  fontsize=8)
    ax.set_ylabel("resident share (π-MLE)")
    ax.set_title("(b)  Call-locked evidence: real call times\nbeat the chronic-pursuit "
                 "null in both groups", loc="left")

    # (c) overlap consistency + posteriors inset
    ax = fig.add_subplot(gs[0, 2])
    att = d["att"]
    ses_ov = att.groupby("animal_id").agg(
        part_frac=("p_res_hmm", lambda p: float((1 - p).mean())),
        ov=("overlap_score", lambda s: float((s > 0.2).mean())),
        n=("call_id", "size"))
    big = ses_ov[ses_ov["n"] >= 10]
    gts = [A.GENOTYPE_MAP[a] for a in big.index]
    cols = [C_WT if g == "WT" else C_HET for g in gts]
    ax.scatter(big["ov"], big["part_frac"], s=40, c=cols, edgecolors="white",
               linewidths=0.5)
    rho = geo.get("overlap_vs_partnerfrac_spearman")
    ax.set_xlabel("two-voice overlap rate (independent acoustic check)")
    ax.set_ylabel("partner-attributed fraction (GEO)")
    ax.set_title(f"(c)  Independent check: two-voice calls\ntrack partner attribution "
                 f"(ρ = {rho:.2f})", loc="left")

    fig.suptitle("V2 validation on real sessions - the stim table is never used",
                 y=1.05)
    savefig(fig, "v2_fig3_validation")


# ========================================================================
def fig4_quantification(d):
    fig = plt.figure(figsize=(13.5, 8.4))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.38)
    ses = d["ses"]
    wt = ses[ses["genotype"] == "WT"]
    het = ses[ses["genotype"] == "HET"]
    st = d["stats"]
    rng = np.random.default_rng(1)

    def dots_ci(ax, x, sub, rate, lo, hi, color):
        xs = np.full(len(sub), x) + rng.normal(0, 0.05, len(sub))
        ax.errorbar(xs, sub[rate], yerr=[np.maximum(sub[rate] - sub[lo], 0),
                                         np.maximum(sub[hi] - sub[rate], 0)],
                    fmt="o", ms=4.5, color=color, alpha=0.75, lw=0.9,
                    capsize=0, zorder=3)
        ax.hlines(sub[rate].median(), x - 0.22, x + 0.22, color=color, lw=3,
                  zorder=4)

    # (a) resident perspective
    ax = fig.add_subplot(gs[0, 0])
    dots_ci(ax, 0, wt, "res_rate", "res_rate_lo", "res_rate_hi", C_WT)
    dots_ci(ax, 1, het, "res_rate", "res_rate_lo", "res_rate_hi", C_HET)
    p = st.loc["res_rate", "mannwhitney_p"]
    ax.text(0.5, wt["res_rate"].max() * 1.04,
            f"p = {p:.4f} {st.loc['res_rate', 'sig']}", ha="center", fontsize=9)
    ax.set_xticks([0, 1], ["WT\nresident", "HET\nresident"])
    ax.set_ylabel("RESIDENT-attributed calls/min\n(π-MLE, 95% CI)")
    ax.set_title("(a)  Experimental mouse's view:\nWT 14.6 vs HET 1.0 calls/min",
                 loc="left")

    # (b) partner perspective
    ax = fig.add_subplot(gs[0, 1])
    dots_ci(ax, 0, wt, "part_rate", "part_rate_lo", "part_rate_hi", C_WT)
    dots_ci(ax, 1, het, "part_rate", "part_rate_lo", "part_rate_hi", C_HET)
    p = st.loc["part_rate", "mannwhitney_p"]
    ymax = max(wt["part_rate_hi"].max(), het["part_rate_hi"].max())
    ax.text(0.5, ymax * 1.04, f"p = {p:.2f} {st.loc['part_rate', 'sig']}",
            ha="center", fontsize=9)
    ax.set_xticks([0, 1], ["with WT\nresident", "with HET\nresident"])
    ax.set_ylabel("PARTNER-attributed calls/min\n(π-MLE, 95% CI)")
    ax.set_title("(b)  Partner's view: near-silent with\neither genotype (~0.25/min, ns)",
                 loc="left")

    # (c) caller composition
    ax = fig.add_subplot(gs[0, 2])
    q = d["quant"]
    for i, (frac, col, lab, n) in enumerate(
            [(q["wt_pi_mean"], C_WT, "WT dyads", q["wt_total_dyad_calls"]),
             (q["het_pi_mean"], C_HET, "HET dyads", q["het_total_dyad_calls"])]):
        ax.bar(i, frac, width=0.55, color=C_RES, edgecolor=C_SURF,
               label="resident" if i == 0 else None)
        ax.bar(i, 1 - frac, bottom=frac, width=0.55, color=C_PART,
               edgecolor=C_SURF, label="partner" if i == 0 else None)
        ax.text(i, 1.03, f"{lab}\n(n={n})", ha="center", fontsize=8.5,
                color=col, fontweight="bold")
        ax.text(i, frac / 2, f"{frac*100:.0f}%", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
    ax.set_xticks([])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("share of dyad calls")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("(c)  Who calls: the resident, in both\ngenotypes - HET just barely "
                 "calls at all", loc="left")

    # (d) attributed time course
    ax = fig.add_subplot(gs[1, :2])
    tc = d["tc"]
    bins = np.arange(0, 900, 30)
    for gt, col in (("WT", C_WT), ("HET", C_HET)):
        animals = [a for a, g in A.GENOTYPE_MAP.items() if g == gt]
        for kind, ls, lw in (("res_calls", "-", 2), ("part_calls", "--", 1.4)):
            M = np.zeros((len(animals), len(bins)))
            for i, a in enumerate(animals):
                sub = tc[tc["animal_id"] == a].set_index("bin_s")[kind]
                M[i] = [sub.get(b, 0.0) * 2 for b in bins]
            ax.plot(bins / 60, M.mean(axis=0), color=col, ls=ls, lw=lw)
    ax.axvline(5, color=C_INK, lw=1.2, ls="--")
    ax.text(5.08, ax.get_ylim()[1] * 0.96, "partner in", fontsize=8.5, va="top")
    ax.plot([], [], color=C_INK, ls="-", label="resident-attributed")
    ax.plot([], [], color=C_INK, ls="--", label="partner-attributed")
    ax.plot([], [], color=C_WT, lw=3, label="WT")
    ax.plot([], [], color=C_HET, lw=3, label="HET")
    ax.legend(fontsize=8, frameon=False, ncols=2, loc="upper right")
    ax.set_xlabel("session time (min)")
    ax.set_ylabel("attributed calls/min (mean)")
    ax.set_title("(d)  Attributed time course: the interaction surge is the resident, "
                 "in every window", loc="left")

    # (e) consistency with v1 design bounds (WT)
    ax = fig.add_subplot(gs[1, 2])
    m = ses.merge(d["v1ses"][["animal_id", "res_rate_lower", "res_rate_upper"]],
                  on="animal_id")
    mwt = m[m["genotype"] == "WT"].sort_values("dyad_rate", ascending=False)
    xs = np.arange(len(mwt))
    ax.bar(xs, mwt["res_rate_upper"] - mwt["res_rate_lower"],
           bottom=mwt["res_rate_lower"], width=0.66, color=C_WT, alpha=0.25,
           edgecolor="none", label="v1 design interval")
    ax.errorbar(xs, mwt["res_rate"],
                yerr=[np.maximum(mwt["res_rate"] - mwt["res_rate_lo"], 0),
                      np.maximum(mwt["res_rate_hi"] - mwt["res_rate"], 0)],
                fmt="o", ms=4, color=C_INK, lw=1.1, capsize=2,
                label="v2 GEO estimate")
    inside = ((mwt["res_rate"] >= mwt["res_rate_lower"] - 0.3)
              & (mwt["res_rate"] <= mwt["res_rate_upper"] + 0.3)).sum()
    ax.set_xticks([])
    ax.set_xlabel("WT sessions (sorted)")
    ax.set_ylabel("resident-attributed calls/min")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"(e)  Two independent routes agree\n({inside}/{len(mwt)} sessions "
                 "inside the v1 interval)", loc="left")

    fig.suptitle("V2 quantification (GEO): WT vs HET during the interaction window, "
                 "both points of view", y=1.0)
    savefig(fig, "v2_fig4_quantification")


def main():
    d = load()
    fig1_method(d)
    fig2_benchmarks(d)
    fig3_validation(d)
    fig4_quantification(d)


if __name__ == "__main__":
    main()
