"""v5 figures: when calls occur, what they do, and why HET mice do not call.

Four multi-panel figures, each written as 300 dpi PNG plus SVG and PDF whose
text stays live and editable in Illustrator or Inkscape.

  fig1  when calls occur: the engagement ladder, proximity, within-bout timing
  fig2  what calls do: directionality and three null results
  fig3  why HET do not call: the decomposition of the rate gap
  fig4  the repertoire control: call shape tracks rate, not genotype

Colour policy: exactly two categorical hues (WT blue, HET orange) taken from a
CVD-validated palette, grey for nulls, red reserved for "this is a null / this
is a reference line". Series colour never encodes rank, only identity.
"""
from __future__ import annotations

import json
import sys
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import attr_common as A
import attr5_core as C

# --- palette --------------------------------------------------------------
WT, HET = "#2a78d6", "#eb6834"          # the two genotypes (categorical)
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"   # text, from strong to faint
SURF, GRID = "#fcfcfb", "#e3e2de"       # page surface and recessive grid
NULLC, CRIT, GOOD = "#b8b6ae", "#d03b3b", "#0ca30c"  # null band, flag, pass
RAMP = ["#86b6ef", "#6da7ec", "#3987e5",             # ordinal ramp, one hue,
        "#2a78d6", "#256abf", "#184f95"]             # used for call types
BIN_MIN = C.BIN_S / 60.0                # minutes per 100 ms bin

mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "font.family": "DejaVu Sans",
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "axes.edgecolor": INK3,
    "axes.linewidth": 0.7, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK, "legend.frameon": False,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3,
    # vector output that stays editable: real glyphs in PDF, live text in SVG
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "pdf.compression": 9, "figure.dpi": 110,
})

PRETTY = {"none": "no social flag", "oriented_toward": "oriented toward",
          "approach": "approach", "nose2nose": "nose-to-nose",
          "nose2body": "body sniff", "nose2anogenital": "anogenital sniff",
          "following": "following", "chasing": "chasing",
          "escape": "escape", "withdrawal_from_partner": "withdrawal",
          "withdrawal_after_contact": "withdrawal after contact",
          "passive": "passive", "rearing": "rearing",
          "sidebyside": "side-by-side", "sidereside": "side-reverse-side",
          "fighting": "fighting"}


def tidy(ax, grid="y"):
    """Strip the top and right spines and push the grid behind the data."""
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)


def save(fig, name):
    """Write one figure three ways: 300 dpi raster plus two vector formats."""
    out = A.ensure_out() / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight",
                pad_inches=0.22)
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", pad_inches=0.22)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    sz = (out / f"{name}.png").stat().st_size / 1024
    print(f"  {name:<26} png {sz:6.0f} KB  + svg + pdf")


# ======================================================== FIGURE 1: WHEN
def fig1(out, bins, sup, why):
    """Calling is a graded index of the caller's own social engagement."""
    fig = plt.figure(figsize=(11.6, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.26,
                          height_ratios=[1.15, 1])

    # --- (a) the engagement ladder ---------------------------------------
    ax = fig.add_subplot(gs[0, :])
    lad = pd.read_csv(out / "v5_ladder.csv")
    order = ["none", "oriented_toward", "approach", "nose2nose", "nose2body",
             "nose2anogenital", "following", "chasing"]
    order = [s for s in order if s in set(lad.state)]
    x = np.arange(len(order))

    # per-session points need a "nothing social" mask built from the raw flags
    b2 = bins.copy()
    b2["animal_id"] = b2["animal_id"].astype(str)
    nonsoc = np.ones(len(b2), bool)
    for f in C.FLAGS:
        nonsoc &= b2[f"m1_{f}"].to_numpy() < 0.5
    b2["__none"] = nonsoc

    for gt, col in (("WT", WT), ("HET", HET)):
        vals = lad[lad.genotype == gt].set_index("state").reindex(order)["rate"]
        ax.plot(x, vals, "-o", color=col, lw=2.0, ms=6, zorder=4,
                label=f"{gt} (n=12)", clip_on=False)
        ids = sorted(b2.loc[b2.genotype == gt, "animal_id"].unique())
        for i, s in enumerate(order):
            m = b2["__none"] if s == "none" else (b2["m1_" + s] > 0.5)
            pts = []
            for a in ids:
                sa = b2.animal_id == a
                nb = int((m & sa).sum())
                if nb >= 60:            # skip states an animal barely entered
                    pts.append(b2.loc[m & sa, "n_calls"].sum() / (nb * BIN_MIN))
            ax.scatter(np.full(len(pts), i) + (0.13 if gt == "HET" else -0.13),
                       np.clip(pts, 0.11, None), s=7, color=col, alpha=0.35,
                       lw=0, zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(s, s) for s in order], rotation=18,
                       ha="right")
    ax.set_ylabel("call rate (calls / min)")
    ax.set_ylim(0.09, 260)
    tidy(ax)
    r_wt = sup["ladder"]["WT"]["rho"]
    r_het = sup["ladder"]["HET"]["rho"]
    ax.set_title("a   Calling is a graded index of social engagement, and the "
                 "gradient is flat in HET", loc="left")
    ax.annotate(f"WT  Spearman rho = {r_wt:+.2f} "
                f"(p = {sup['ladder']['WT']['p']:.3f})",
                (0.015, 0.93), xycoords="axes fraction", color=WT,
                fontsize=8.5, fontweight="bold")
    ax.annotate(f"HET rho = {r_het:+.2f} "
                f"(p = {sup['ladder']['HET']['p']:.2f}), flat",
                (0.015, 0.845), xycoords="axes fraction", color=HET,
                fontsize=8.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.annotate("dots = individual sessions; a session with no calls in a "
                "state is drawn on the axis floor",
                (0.015, 0.035), xycoords="axes fraction", fontsize=6.8,
                color=INK3)

    # --- (b) proximity gain ----------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    g = pd.read_csv(out / "v5_distance_gain.csv")
    for gt, col in (("WT", WT), ("HET", HET)):
        d = g[g.genotype == gt].sort_values("dist_med")
        sl = why["gain"][f"slope_{gt}"]
        ax.plot(d.dist_med, np.clip(d.rate, 0.2, None), "-o", color=col,
                lw=1.8, ms=5, label=f"{gt}   slope {sl:+.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("inter-animal distance (px)")
    ax.set_ylabel("call rate (calls / min)")
    sw, sh = why["gain"]["slope_WT"], why["gain"]["slope_HET"]
    ci = why["gain"]["slope_diff_ci"]
    ax.set_title("b   WT call more the closer they are; HET do not", loc="left")
    ax.annotate(f"difference {sw - sh:+.2f}\n"
                f"[{ci[0]:+.2f}, {ci[1]:+.2f}], p < 0.001",
                (0.97, 0.93), xycoords="axes fraction", ha="right", va="top",
                fontsize=7.6, color=INK2)
    ax.legend(loc="lower left", fontsize=7.8)
    tidy(ax, "both")

    # --- (c) within-bout timing ------------------------------------------
    # Pooled to 0.3 s bins with exact Poisson (chi-square) confidence limits,
    # otherwise the per-100 ms rates are too noisy to read.
    ax = fig.add_subplot(gs[1, 1])
    bt = pd.read_csv(out / "v5_bout_timing.csv")
    k = 3
    n = (len(bt) // k) * k
    el = bt.elapsed_s.to_numpy()[:n].reshape(-1, k).mean(1)
    ex = bt.exposure_bins.to_numpy()[:n].reshape(-1, k).sum(1)
    counts = (bt.rate.to_numpy()[:n] * bt.exposure_bins.to_numpy()[:n]
              * BIN_MIN).reshape(-1, k).sum(1)
    expo_min = np.maximum(ex * BIN_MIN, 1e-9)
    rr = counts / expo_min
    lo_ = stats.chi2.ppf(0.025, 2 * counts) / 2 / expo_min
    hi_ = stats.chi2.ppf(0.975, 2 * (counts + 1)) / 2 / expo_min
    ax.fill_between(el, lo_, hi_, color=WT, alpha=0.16, lw=0)
    ax.plot(el, rr, "-o", color=WT, lw=2.0, ms=4)
    ax.axhline(rr.mean(), color=INK3, lw=0.9, ls=(0, (4, 3)))
    ax.annotate("bout mean", (el.max(), rr.mean()), xytext=(-4, 5),
                textcoords="offset points", ha="right", color=INK2,
                fontsize=7.5)
    ax.set_xlabel("time since onset of a social-investigation bout (s)")
    ax.set_ylabel("call rate (calls / min)")
    ax.set_ylim(0, float(np.nanmax(hi_)) * 1.22)
    ax.set_title("c   Calls build up inside a bout, no onset transient",
                 loc="left")
    ax.annotate(f"first 0.5 s {sup['timing']['rate_early']:.0f}/min  vs  "
                f"later {sup['timing']['rate_late']:.0f}/min: calling RISES\n"
                f"as the bout runs on (rho = {sup['timing']['rho']:+.2f}, "
                f"p = {sup['timing']['p']:.2f}); a phasic response to the\n"
                f"onset would instead spike and decay",
                (0.03, 0.97), xycoords="axes fraction", va="top",
                fontsize=7.2, color=INK2)
    tidy(ax)

    fig.suptitle("When do mice call?  Calling indexes the intensity of the "
                 "resident's own social engagement",
                 x=0.011, y=1.005, ha="left", fontsize=12.5, fontweight="bold")
    save(fig, "v5_fig1_when")


# ======================================================== FIGURE 2: WHAT
def fig2(out, direc, haz, perm):
    """Directionality plus the three null results on call consequences."""
    comp = json.load(open(out / "v5_composite_direction.json"))
    R, P = comp["results"], comp["profiles"]
    fig = plt.figure(figsize=(12.4, 8.4))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.46,
                          width_ratios=[1.0, 1.0, 1.02])

    # --- (a) call-triggered behaviour profile (the even part dominates) ---
    ax = fig.add_subplot(gs[0, 0])
    p = P["resident_investigation"]
    lags = np.array(p["lags"])
    tt = np.concatenate([-lags[::-1], [0], lags])
    yy = np.concatenate([np.array(p["before"])[::-1], [np.nan],
                         np.array(p["after"])])
    ax.plot(tt, yy, color=WT, lw=1.9)
    ax.fill_between(tt, p["occupancy"], yy, color=WT, alpha=0.12, lw=0)
    ax.axhline(p["occupancy"], color=INK3, lw=0.9, ls=(0, (4, 3)))
    ax.axvline(0, color=INK3, lw=0.9)
    ax.set_xlabel("time relative to a call (s)")
    ax.set_ylabel("P(resident investigating)")
    ax.set_title("a   Calls sit inside investigation bouts", loc="left")
    ax.annotate("session baseline", (lags.max(), p["occupancy"]),
                xytext=(-3, -11), textcoords="offset points", ha="right",
                color=INK2, fontsize=7.2)
    ax.annotate(f"{R['resident_investigation']['enrichment']:.2f}x enriched,\n"
                "but tilted: higher BEFORE the call",
                (0.03, 0.96), xycoords="axes fraction", va="top",
                fontsize=7.5, color=INK2)
    tidy(ax)

    # --- (b) the odd component against its exact circular-shift null ------
    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(lags, p["odd_lo"], p["odd_hi"], color=NULLC, alpha=0.55,
                    lw=0, label="95% circular-shift null (pointwise)")
    ax.plot(lags, p["odd"], color=WT, lw=1.9, label="observed")
    ax.axhline(0, color=INK3, lw=0.9)
    ax.set_xlabel("lag tau (s)")
    ax.set_ylabel("P(tau) - P(-tau)")
    r = R["resident_investigation"]
    ax.set_title("b   The odd part is negative: behaviour first", loc="left")
    ax.legend(loc="lower left", fontsize=6.8)
    ax.annotate("call leads ->", (0.97, 0.95), xycoords="axes fraction",
                ha="right", fontsize=7.2, color=INK2)
    ax.annotate("<- behaviour leads", (0.97, 0.30), xycoords="axes fraction",
                ha="right", fontsize=7.2, color=INK2)
    ax.annotate(f"Lambda = {r['lambda']:+.2f},  p = {r['p']:.3f}",
                (0.97, 0.86), xycoords="axes fraction", ha="right",
                fontsize=7.8, color=WT, fontweight="bold")
    tidy(ax)

    # --- (c) the four pre-specified suites --------------------------------
    ax = fig.add_subplot(gs[0, 2])
    names = ["resident_investigation", "partner_investigation",
             "partner_flight", "social_contact"]
    disp = ["resident's own\ninvestigation",
            "partner investigating\nthe resident",
            "partner's\nflight", "social\ncontact"]
    yy = np.arange(len(names))[::-1]
    for y_, nm in zip(yy, names):
        rr_ = R[nm]
        sd = rr_["lam_null_sd"]
        ax.barh(y_, 2 * 1.96 * sd, left=rr_["lam_null_mean"] - 1.96 * sd,
                height=0.52, color=NULLC, alpha=0.55, lw=0, zorder=1)
        sig = rr_["q"] < 0.05
        ax.plot(rr_["lambda"], y_, "o", ms=8 if sig else 6,
                color=WT if sig else INK3, zorder=3)
        ax.annotate(f"p = {rr_['p']:.3f}" + ("  *" if sig else ""),
                    (rr_["lambda"], y_), xytext=(0, -17),
                    textcoords="offset points", ha="center", fontsize=7.2,
                    color=WT if sig else INK2,
                    fontweight="bold" if sig else "normal")
    ax.axvline(0, color=INK3, lw=0.9)
    ax.set_yticks(yy)
    ax.set_yticklabels(disp, fontsize=7.4)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("Lambda   <- behaviour leads    call leads ->")
    ax.set_title("c   Only the caller's own behaviour leads", loc="left")
    ax.tick_params(axis="y", length=0)
    ax.annotate("grey = 95% null", (0.03, 0.04), xycoords="axes fraction",
                fontsize=7, color=INK2)
    tidy(ax, "x")

    # --- (d, e) the two hazard directions, same estimator ------------------
    for j, (mod, title) in enumerate([
            ("A_antecedent", "d   Does a partner action trigger a call?"),
            ("B_consequence", "e   Does a call trigger a behaviour?")]):
        ax = fig.add_subplot(gs[1, j])
        h = haz[(haz["model"] == mod) & (haz["set"] == "WT")].copy()
        h["lab"] = (h["who"].map({"m1": "res", "m2": "ptn"}) + " "
                    + h["behavior"].map(lambda b: PRETTY.get(b, b)))
        h = h.sort_values("or")
        y2 = np.arange(len(h))
        ax.hlines(y2, h["lo"].clip(0.05, 20), h["hi"].clip(0.05, 20),
                  color=INK3, lw=1.1)
        ax.plot(h["or"], y2, "o", color=WT, ms=4.0, zorder=3)
        ax.axvline(1, color=CRIT, lw=1.0, ls=(0, (4, 3)))
        ax.set_xscale("log")
        ax.set_xlim(0.045, 22)
        ax.set_yticks(y2)
        ax.set_yticklabels(h["lab"], fontsize=5.8)
        ax.set_ylim(-1.4, len(h) - 0.4)
        ax.set_xlabel("odds ratio (95% CI)")
        ax.set_title(title, loc="left")
        ax.tick_params(axis="y", length=0)
        ax.annotate(f"{(h['q'] < 0.05).sum()} of {len(h)} survive FDR",
                    (0.5, 0.015), xycoords="axes fraction", ha="center",
                    fontsize=7.6, color=CRIT, fontweight="bold")
        tidy(ax, "x")

    # --- (f) does call type matter? block-permutation null -----------------
    ax = fig.add_subplot(gs[1, 2])
    lbl = {"d_app_m2": "partner approach", "d_app_m1": "resident approach",
           "d_speed_m2": "partner speed", "d_speed_m1": "resident speed",
           "d_dist": "distance", "esc_post": "partner escape",
           "wdr_post": "partner withdrawal"}
    pm = perm.copy().sort_values("obs_t")
    y3 = np.arange(len(pm))
    ax.barh(y3, pm["null_mean_t"], color=NULLC, height=0.62, lw=0,
            label="null mean |t|")
    ax.plot(pm["obs_t"], y3, "D", color=WT, ms=5, zorder=3,
            label="observed |t|")
    ax.set_yticks(y3)
    ax.set_yticklabels([lbl.get(o, o) for o in pm["outcome"]], fontsize=7)
    ax.set_xlabel("|t|,  long vs ultrashort calls")
    ax.set_xlim(0, max(pm["obs_t"]) * 1.55)
    ax.set_ylim(-1.5, len(pm) - 0.4)
    ax.set_title("f   Does call TYPE change what follows?", loc="left")
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper right", fontsize=7)
    ax.annotate(f"block-permutation q >= {perm['q_perm'].min():.2f} everywhere",
                (0.5, 0.015), xycoords="axes fraction", ha="center",
                fontsize=7.6, color=CRIT, fontweight="bold")
    tidy(ax, "x")

    fig.suptitle("What do calls do?  The caller's own behaviour leads the "
                 "call; nothing reliably follows it",
                 x=0.011, y=1.005, ha="left", fontsize=12.5, fontweight="bold")
    save(fig, "v5_fig2_what")


# ======================================================== FIGURE 3: WHY
def fig3(out, why, sup, bouts, lat, esc):
    """The HET deficit decomposed: not opportunity, not the larynx, the gain."""
    fig = plt.figure(figsize=(13.2, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.60, wspace=0.42)
    k = why["kitagawa"]

    # --- (a) Kitagawa waterfall -------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    vals = [k["R_HET"], k["opportunity"], k["propensity"]]
    labs = ["HET\nobserved", "+opportunity\n(10%)", "+propensity\n(90%)"]
    bottoms = [0, k["R_HET"], k["R_HET"] + k["opportunity"]]
    cols = [HET, INK3, WT]
    for i, (v, b, c) in enumerate(zip(vals, bottoms, cols)):
        ax.bar(i, v, bottom=b, color=c, width=0.62, lw=0)
    ax.bar(3, k["R_WT"], color=WT, width=0.62, lw=0)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labs + ["WT\nobserved"], fontsize=7.0)
    ax.set_ylabel("calls / min")
    ci = k["pct_propensity_ci"]
    ax.set_title("a   The gap is propensity, not opportunity", loc="left")
    ax.annotate(f"CI\n{ci[0]:.0f}-{ci[1]:.0f}%",
                (2, k["R_HET"] + k["opportunity"] + k["propensity"] / 2),
                ha="center", va="center", color="white", fontweight="bold",
                fontsize=7.6)
    ax.annotate(f"HET given WT's exact time budget:\n"
                f"{k['cf_HET_with_WT_time']:.2f} calls/min "
                f"(vs {k['R_HET']:.2f} observed), no change",
                (0.5, -0.32), xycoords="axes fraction", ha="center",
                va="top", fontsize=7.4, color=CRIT, fontweight="bold")
    tidy(ax)

    # --- (b) time budget against rate, state by state ----------------------
    ax = fig.add_subplot(gs[0, 1])
    st = k["states"]
    tw, th = np.array(k["T_WT"]), np.array(k["T_HET"])
    lw_, lh_ = np.array(k["lam_WT"]), np.array(k["lam_HET"])
    ax.scatter(100 * tw, np.clip(lw_, 0.3, None), s=64, color=WT, zorder=3,
               label="WT", lw=0)
    ax.scatter(100 * th, np.clip(lh_, 0.3, None), s=64, color=HET, zorder=3,
               label="HET", lw=0)
    for i, s in enumerate(st):
        ax.plot([100 * tw[i], 100 * th[i]],
                [max(lw_[i], .3), max(lh_[i], .3)], color=INK3, lw=0.8,
                zorder=1)
        ax.annotate(s, (100 * tw[i], max(lw_[i], .3)), xytext=(5, 4),
                    textcoords="offset points", fontsize=6.8, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("% of the interaction window spent in the state")
    ax.set_ylabel("call rate in that state (calls / min)")
    ax.set_title("b   Similar time budgets, different rates", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    ax.annotate("each grey line joins the same state in the two genotypes:\n"
                "they move mostly VERTICALLY, a rate change, not a\n"
                "behaviour change",
                (0.5, -0.32), xycoords="axes fraction", va="top", ha="center",
                fontsize=7.4, color=INK2)
    tidy(ax, "both")

    # --- (c) social gain ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    sg = why["social_gain"]
    for i, (gt, col, g_, ci_) in enumerate([
            ("WT", WT, sg["WT_gain"], sg["WT_gain_ci"]),
            ("HET", HET, sg["HET_gain"], sg["HET_gain_ci"])]):
        ax.bar(i, g_, color=col, width=0.5, lw=0)
        ax.errorbar(i, g_, yerr=[[g_ - ci_[0]], [ci_[1] - g_]], color=INK,
                    lw=1.2, capsize=4)
        ax.annotate(f"{g_:.2f}x", (i, ci_[1]), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=9,
                    fontweight="bold", color=col)
    ax.axhline(1, color=CRIT, lw=1.0, ls=(0, (4, 3)))
    ax.annotate("no modulation", (1.42, 1), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=7.2,
                color=CRIT)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["WT", "HET"])
    ax.set_ylabel("call rate  social / non-social")
    ax.set_title("c   Social gain is abolished", loc="left")
    ax.annotate(f"ratio {sg['gain_ratio']:.2f}x "
                f"[{sg['gain_ratio_ci'][0]:.1f}, {sg['gain_ratio_ci'][1]:.1f}], "
                f"bootstrap p < 0.001\nleave-one-session-out (WT): "
                f"{sup['loso_gain']['min']:.1f}-{sup['loso_gain']['max']:.1f}x",
                (0.5, -0.32), xycoords="axes fraction", ha="center", va="top",
                fontsize=7.4, color=INK2)
    ax.set_ylim(0, max(sg["WT_gain_ci"][1], 1.4) * 1.28)
    tidy(ax)

    # --- (d) latency to the first call -------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for gt, col in (("WT", WT), ("HET", HET)):
        v = np.sort(lat[lat.genotype == gt].latency_s.to_numpy())
        y = 1 - np.arange(len(v)) / len(v)
        ax.step(np.concatenate([[0], v]), np.concatenate([[1], y]),
                where="post", color=col, lw=2.0, label=gt)
    ax.set_xlabel("time after the partner is introduced (s)")
    ax.set_ylabel("fraction not yet calling")
    ax.set_xlim(0, 200)
    ax.set_title("d   HET start calling just as fast", loc="left")
    ax.annotate(f"median {why['latency']['WT_median']:.0f} s vs "
                f"{why['latency']['HET_median']:.0f} s\n"
                f"log-rank p = {why['latency']['p']:.2f}",
                (0.97, 0.92), xycoords="axes fraction", ha="right", va="top",
                fontsize=7.6, color=INK2)
    ax.legend(loc="center right", fontsize=8)
    tidy(ax)

    # --- (e) rate = initiation x maintenance -------------------------------
    ax = fig.add_subplot(gs[1, 1])
    comps = [("bouts_per_min", "bouts\nper min\n(initiation)"),
             ("calls_per_bout", "calls\nper bout\n(maintenance)"),
             ("rate", "calls\nper min\n(product)")]
    w = 0.34
    jitter = np.random.default_rng(3)
    for i, (col_, lab) in enumerate(comps):
        for j, (gt, col) in enumerate((("WT", WT), ("HET", HET))):
            v = bouts.loc[bouts.genotype == gt, col_].dropna()
            ax.bar(i + (j - 0.5) * w, v.median(), width=w * 0.92, color=col,
                   lw=0, zorder=2)
            ax.scatter(np.full(len(v), i + (j - 0.5) * w)
                       + jitter.normal(0, 0.03, len(v)), v, s=8, color=INK,
                       alpha=0.45, lw=0, zorder=3)
        wv = bouts.loc[bouts.genotype == "WT", col_].median()
        hv = bouts.loc[bouts.genotype == "HET", col_].median()
        ax.annotate(f"{wv / max(hv, 1e-9):.1f}x", (i, max(wv, hv) * 1.5),
                    ha="center", fontsize=8, fontweight="bold", color=INK)
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels([c[1] for c in comps], fontsize=7.4)
    ax.set_ylabel("per session (median, log scale)")
    ax.set_title("e   Starting and sustaining both reduced", loc="left")
    ax.legend(handles=[Line2D([], [], color=WT, lw=6, label="WT"),
                       Line2D([], [], color=HET, lw=6, label="HET")],
              loc="upper center", fontsize=8, ncol=2)
    ax.set_ylim(0.18, 260)
    tidy(ax)

    # --- (f) how far a calling bout gets -----------------------------------
    ax = fig.add_subplot(gs[1, 2])
    for j, (gt, col) in enumerate((("WT", WT), ("HET", HET))):
        v = esc[esc.genotype == gt]
        ax.bar(np.arange(2) + (j - 0.5) * 0.34,
               [v.p_ge2.median(), v.p_ge4.median()], width=0.31, color=col,
               lw=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["bout reaches\n>= 2 calls", "bout reaches\n>= 4 calls"],
                       fontsize=7.6)
    ax.set_ylabel("fraction of calling bouts")
    ax.set_title("f   HET almost never escalate", loc="left")
    e2 = sup["escalation_bout"]
    ax.annotate(f"{e2['p_ge2']['WT']:.2f} vs {e2['p_ge2']['HET']:.2f}\n"
                f"p = {e2['p_ge2']['p']:.3f}", (0, e2["p_ge2"]["WT"]),
                xytext=(0, 7), textcoords="offset points", ha="center",
                fontsize=7.4, color=INK2)
    ax.annotate(f"{e2['p_ge4']['WT']:.2f} vs {e2['p_ge4']['HET']:.2f}\n"
                f"p = {e2['p_ge4']['p']:.3f}", (1, e2["p_ge4"]["WT"]),
                xytext=(0, 7), textcoords="offset points", ha="center",
                fontsize=7.4, color=INK2)
    ax.set_ylim(0, 0.60)
    ax.legend(handles=[Line2D([], [], color=WT, lw=6, label="WT"),
                       Line2D([], [], color=HET, lw=6, label="HET")],
              loc="upper center", fontsize=8, ncol=2)
    tidy(ax)

    fig.suptitle("Why don't HET mice call?  Not opportunity, not the larynx, "
                 "the gain from social engagement to voice",
                 x=0.011, y=1.005, ha="left", fontsize=12.5, fontweight="bold")
    save(fig, "v5_fig3_why")


# =================================================== FIGURE 4: REPERTOIRE
def fig4(out, calls, prof, rec, why):
    """The apparent HET acoustic phenotype is a calling-rate effect."""
    fig = plt.figure(figsize=(12.8, 3.8))
    gs = fig.add_gridspec(1, 4, wspace=0.40)
    names = list(prof.sort_values("rf_dur_ms")["name"])
    cmap = dict(zip(names, RAMP))

    # --- (a) the six types in duration x FM-slope space --------------------
    ax = fig.add_subplot(gs[0, 0])
    for nm in names:
        d = calls[calls.ctype_name == nm]
        ax.scatter(d.rf_dur_ms, d.rf_slope_khz_ms, s=4, color=cmap[nm],
                   alpha=0.5, lw=0)
    for nm in names:
        d = calls[calls.ctype_name == nm]
        mx, my = d.rf_dur_ms.median(), d.rf_slope_khz_ms.median()
        ax.plot(mx, my, "o", ms=9, mfc=cmap[nm], mec="white", mew=1.4,
                zorder=5)
        ax.annotate(nm, (mx, my), xytext=(0, 12 if my >= 0 else -18),
                    textcoords="offset points", ha="center", fontsize=6.2,
                    color=INK, fontweight="bold", zorder=6)
    ax.set_xscale("log")
    ax.set_xlabel("duration (ms)")
    ax.set_ylabel("FM slope (kHz / ms)")
    ax.set_ylim(-8, 8)
    ax.axhline(0, color=INK3, lw=0.7)
    ax.set_title("a   Six call types, BIC-chosen", loc="left")
    tidy(ax, "both")

    # --- (b) composition by genotype ---------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    comp = pd.crosstab(calls.ctype_name, calls.genotype, normalize="columns")
    comp = comp.reindex(names)
    bot = np.zeros(2)
    for nm in names:
        v = comp.loc[nm, ["WT", "HET"]].to_numpy(float)
        ax.bar([0, 1], v, bottom=bot, color=cmap[nm], width=0.6, lw=0.6,
               edgecolor=SURF)
        for i in range(2):
            if v[i] > 0.08:
                ax.annotate(nm.replace(" ", "\n"), (i, bot[i] + v[i] / 2),
                            ha="center", va="center", fontsize=5.8,
                            color="white" if nm in names[3:] else INK)
        bot += v
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["WT", "HET"])
    ax.set_ylabel("share of calls")
    ax.set_title("b   HET calls look different...", loc="left")
    tidy(ax)

    # --- (c) the control that reinterprets (b) -----------------------------
    ax = fig.add_subplot(gs[0, 2])
    for gt, col in (("WT", WT), ("HET", HET)):
        d = rec[rec.genotype == gt]
        ax.scatter(d.n, d.fr_ultra, s=34, color=col, lw=0, label=gt, zorder=3)
    xx = np.logspace(np.log10(2), np.log10(750), 60)
    m = np.polyfit(np.log10(rec[rec.genotype == "WT"].n),
                   rec[rec.genotype == "WT"].fr_ultra, 1)
    ax.plot(xx, np.clip(np.polyval(m, np.log10(xx)), 0, 1), color=INK3,
            lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("calls in the session")
    ax.set_ylabel("fraction ultrashort (< 9 ms)")
    ax.set_title("c   ...because they call less", loc="left")
    ax.set_xticks([1, 10, 100, 1000])
    ax.annotate(f"within WT alone rho = {why['recruitment']['rho_WT']:+.2f} "
                f"(p = {why['recruitment']['p_rho_WT']:.3f});\n"
                f"residual genotype offset ns (p = "
                f"{why['recruitment']['p_genotype']:.2f}); rate-matched WT\n"
                f"sessions indistinguishable (p = "
                f"{why['recruitment']['ratematched_p']:.2f})",
                (0.5, -0.36), xycoords="axes fraction", va="top", ha="center",
                fontsize=6.6, color=INK2)
    ax.legend(loc="upper right", fontsize=8)
    tidy(ax, "both")

    # --- (d) what is actually missing: the upper tail ----------------------
    ax = fig.add_subplot(gs[0, 3])
    for gt, col in (("WT", WT), ("HET", HET)):
        v = np.sort(rec[rec.genotype == gt].n.to_numpy())[::-1]
        ax.plot(np.arange(1, len(v) + 1), v, "-o", color=col, ms=4, lw=1.5,
                label=gt)
    ax.set_yscale("log")
    ax.axhline(30, color=CRIT, lw=1.0, ls=(0, (4, 3)))
    ax.annotate("30 calls", (12, 32), ha="right", fontsize=7, color=CRIT)
    ax.set_xlabel("session rank within genotype")
    ax.set_xticks([1, 4, 8, 12])
    ax.set_ylabel("calls in the session")
    ax.set_title("d   The tail is what is missing", loc="left")
    ax.annotate("8/12 WT exceed 30 calls\n0/12 HET  (p = 0.001)",
                (0.96, 0.99), xycoords="axes fraction", ha="right", va="top",
                fontsize=7.3, color=INK2)
    ax.legend(loc="lower left", fontsize=8)
    tidy(ax)

    fig.suptitle("The HET repertoire is not abnormal, it is what any mouse's "
                 "repertoire looks like at a low calling rate",
                 x=0.011, y=1.06, ha="left", fontsize=12.5, fontweight="bold")
    save(fig, "v5_fig4_repertoire")


def main():
    out = A.ensure_out()
    bins = C.load_bins()
    why = json.load(open(out / "v5_why_het.json"))
    sup = json.load(open(out / "v5_supplement.json"))
    direc = pd.read_csv(out / "v5_directionality.csv")
    haz = pd.read_csv(out / "v5_hazard.csv")
    perm = pd.read_csv(out / "v5_calltype_permutation.csv")
    bouts = pd.read_csv(out / "v5_bouts.csv")
    lat = pd.read_csv(out / "v5_latency.csv")
    esc = pd.read_csv(out / "v5_bout_escalation.csv")
    calls = pd.read_csv(out / "v5_calls.csv", low_memory=False)
    cprof = pd.read_csv(out / "v5_calltype_profile.csv")
    rec = pd.read_csv(out / "v5_recruitment.csv")

    fig1(out, bins, sup, why)
    fig2(out, direc, haz, perm)
    fig3(out, why, sup, bouts, lat, esc)
    fig4(out, calls, cprof, rec, why)


if __name__ == "__main__":
    main()
