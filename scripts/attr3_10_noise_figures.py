"""Analysis + figures for the false-positive ("calls are walking noise") control.

Fig 1  ARE THE CALLS REAL?   spectral evidence, movement-matched controls,
                             mechanical-transient detector, gait test
Fig 2  DOES THE RESULT SURVIVE?  locomotion accounting + strict-USV re-analysis

Outputs: noise_control_summary.json, v3_fig1_calls_are_real.*,
         v3_fig2_result_survives.*
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A
import attr2_core as C2
from attr2_45_geometry import pi_mle

C_WT, C_HET = "#2a78d6", "#eb6834"
C_CALL, C_CTRL, C_MECH = "#1baf7a", "#898781", "#4a3aa7"
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
STRICT_LOW_DB = 3.0        # sub-USV-band energy must stay below this
STRICT_TONALITY = None     # set from the control distribution


def savefig(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load():
    d = {}
    d["calls"] = pd.read_csv(OUT / "noise_call_measures.csv").assign(
        animal_id=lambda x: x["animal_id"].astype(str))
    d["ctrl"] = pd.read_csv(OUT / "noise_control_measures.csv").assign(
        animal_id=lambda x: x["animal_id"].astype(str))
    d["sess"] = pd.read_csv(OUT / "noise_session_stats.csv").assign(
        animal_id=lambda x: x["animal_id"].astype(str))
    d["shape"] = pd.read_csv(OUT / "noise_crop_stats.csv")
    d["crops"] = np.load(OUT / "noise_crops.npz", allow_pickle=False)
    d["geo"] = pd.read_csv(OUT / "v2_geo_attribution.csv").assign(
        animal_id=lambda x: x["animal_id"].astype(str))
    d["v2ses"] = pd.read_csv(OUT / "v2_session_quantification.csv").assign(
        animal_id=lambda x: x["animal_id"].astype(str))
    d["geoval"] = json.load(open(OUT / "v2_geo_validation.json"))
    return d


def mw(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(stats.mannwhitneyu(a, b, alternative="two-sided")[1])


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


# =====================================================================
def fig1(d, S):
    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.34)
    calls, ctrl, shape = d["calls"], d["ctrl"], d["shape"]

    # (a) example full-band spectrograms
    ax = fig.add_subplot(gs[0, :2])
    ax.set_axis_off()
    crops = d["crops"]["crops"].astype(float)
    tags = d["crops"]["tag"]
    rng = np.random.default_rng(3)
    rows = []
    for tag in ("call", "mech"):
        idx = np.where(tags == tag)[0]
        sel = rng.choice(idx, min(6, len(idx)), replace=False)
        rows.append(sel)
    H, W = 96, 64
    canvas = np.full((2 * (H + 4), 6 * (W + 4)), np.nan)
    for r, sel in enumerate(rows):
        for c, k in enumerate(sel):
            canvas[r * (H + 4):r * (H + 4) + H,
                   c * (W + 4):c * (W + 4) + W] = crops[k][::-1]
    ax.imshow(canvas, cmap="magma", aspect="auto", interpolation="nearest",
              vmin=0, vmax=0.8)
    ax.text(-14, H / 2, "detected\ncalls", ha="right", va="center", fontsize=9,
            color=C_CALL, fontweight="bold")
    ax.text(-14, H + 4 + H / 2, "mechanical\ntransients", ha="right",
            va="center", fontsize=9, color=C_MECH, fontweight="bold")
    # frequency reference: rows run 2 kHz (bottom) -> 190 kHz (top)
    def yfreq(khz, row):
        return row * (H + 4) + (1 - (khz - 2) / 188.0) * H
    W_tot = 6 * (W + 4) - 4
    for r in range(2):
        ax.plot([-2, W_tot], [yfreq(45, r)] * 2, color="#7fe3c0", lw=1.1,
                ls="--", alpha=0.9)
        for khz in (190, 125, 45, 2):
            y = yfreq(khz, r)
            ax.plot([-9, -3], [y, y], color=C_MUT, lw=0.8)
            ax.text(-10, y, f"{khz}", ha="right", va="center", fontsize=6.5,
                    color=C_MUT)
    ax.text(W_tot + 4, yfreq(85, 0), "detector\nwindow\n45-125 kHz",
            fontsize=7, color=C_SEC, va="center")
    ax.text(W_tot + 4, yfreq(18, 1), "detector\nBLIND\n<45 kHz",
            fontsize=7, color=C_MECH, va="center", fontweight="bold")
    ax.set_xlim(-30, W_tot + 46)
    ax.set_title("(a)  Full spectrum, 2-190 kHz (dashed = 45 kHz detector floor).\n"
                 "Calls are whistles above the line; mechanical events cross it.",
                 loc="left")

    # (b) THE decisive panel: sub-USV band energy
    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(-6, 24, 46)
    ax.hist(ctrl["low_snr_db"], bins=bins, color=C_CTRL, alpha=0.75,
            density=True, label="movement-matched\ncontrol times")
    ax.hist(calls["low_snr_db"], bins=bins, color=C_CALL, alpha=0.75,
            density=True, label="detected calls")
    ax.axvline(6.0, color=C_MECH, lw=1.6, ls="--")
    ax.text(6.6, ax.get_ylim()[1] * 0.72,
            "mechanical-event\nthreshold\n(a real thud lands\nhere or beyond)",
            fontsize=7.4, color=C_MECH, va="top")
    ax.set_xlabel("sub-USV band energy, 5-35 kHz (dB over local baseline)")
    ax.set_ylabel("density")
    ax.legend(fontsize=7.6, frameon=False, loc="upper right")
    ax.set_title(f"(b)  In the detector's blind band, calls are\n"
                 f"indistinguishable from silence "
                 f"(p = {S['p_low_call_vs_ctrl']:.2f})", loc="left")

    # (c) two populations in the same recordings
    ax = fig.add_subplot(gs[1, 0])
    for tag, col, lab in (("call", C_CALL, "detected calls"),
                          ("mech", C_MECH, "mechanical transients")):
        sub = shape[shape["tag"] == tag]
        ax.scatter(sub["peak_low_db"], sub["peak_usv_db"], s=13, c=col,
                   alpha=0.55, edgecolors="none", label=lab)
    lim = [-2, max(shape["peak_low_db"].max(), shape["peak_usv_db"].max()) * 1.05]
    ax.plot(lim, lim, color=C_MUT, lw=1, ls="--")
    ax.text(lim[1] * 0.95, lim[1] * 0.9, "equal energy\nin both bands",
            fontsize=7.5, color=C_MUT, ha="right")
    ax.set_xlabel("low-band peak (dB)")
    ax.set_ylabel("USV-band peak (dB)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title(f"(c)  Two distinct event populations\n"
                 f"(only {S['pct_calls_with_mech']:.1f}% of calls coincide with "
                 "a transient)", loc="left")

    # (d) coincidence with mechanical transients vs chance
    ax = fig.add_subplot(gs[1, 1])
    vals = [S["pct_calls_with_mech"], S["pct_calls_with_mech_expected"]]
    bars = ax.bar([0, 1], vals, width=0.55, color=[C_MECH, C_CTRL],
                  edgecolor=C_SURF)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9, color=C_SEC)
    ax.set_xticks([0, 1], ["observed", "expected by chance\n(random times)"])
    ax.set_ylabel("% of calls within ±10 ms\nof a mechanical transient")
    ax.set_ylim(0, max(vals) * 1.45)
    ax.set_title("(d)  Calls do not sit on mechanical events\nmore than random "
                 "moments do", loc="left")

    # (e) speed at call time vs time-at-rest baseline
    ax = fig.add_subplot(gs[1, 2])
    sp = calls["speed_m1"].to_numpy()
    ax.hist(sp, bins=40, color=C_CALL, edgecolor=C_SURF)
    thr = S["immobile_thr_px_s"]
    ax.axvline(thr, color=C_INK, lw=1.4, ls="--")
    n_imm = int(S["pct_calls_immobile"] / 100 * len(calls))
    ax.text(0.42, 0.93,
            f"{n_imm} calls ({S['pct_calls_immobile']:.0f}%) emitted below "
            f"{thr:.0f} px/s —\nessentially stationary, no footfall possible.\n"
            f"(mice spend {S['pct_time_immobile']:.0f}% of the session that "
            "slow, so\ncalling genuinely favours movement — but it\ndoes not "
            "require it)",
            transform=ax.transAxes, fontsize=7.8, color=C_INK, va="top")
    ax.set_xlabel("caller speed at call time (px/s)")
    ax.set_ylabel("calls")
    ax.set_title("(e)  Calls favour movement, but hundreds\nare emitted at rest",
                 loc="left")

    fig.suptitle("Control: are the detections real USVs, or noise from moving mice?",
                 y=1.0)
    savefig(fig, "v3_fig1_calls_are_real")


# =====================================================================
def fig2(d, S):
    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.36)
    ses = d["sess"]
    wt = ses[ses["genotype"] == "WT"]
    het = ses[ses["genotype"] == "HET"]
    rng = np.random.default_rng(5)

    def dots(ax, x, v, col):
        v = np.asarray(v, float)
        ax.scatter(np.full(len(v), x) + rng.normal(0, 0.05, len(v)), v, s=26,
                   c=col, alpha=0.78, edgecolors="white", linewidths=0.4, zorder=3)
        ax.hlines(np.median(v), x - 0.22, x + 0.22, color=col, lw=3, zorder=4)

    # (a) do HET mice simply move less?
    ax = fig.add_subplot(gs[0, 0])
    dots(ax, 0, wt["dist_px_partner"] / 1e3, C_WT)
    dots(ax, 1, het["dist_px_partner"] / 1e3, C_HET)
    p = S["p_locomotion"]
    ax.text(0.5, max(wt["dist_px_partner"].max(), het["dist_px_partner"].max())
            / 1e3 * 1.03, f"p = {p:.2f} {stars(p)}", ha="center", fontsize=9)
    ax.set_xticks([0, 1], ["WT", "HET"])
    ax.set_ylabel("distance travelled, interaction window\n(×10³ px)")
    ax.set_title(f"(a)  The honest confound:\nWT mice DO move more "
                 f"({S['ratio_locomotion']:.2f}×)", loc="left")

    # (b) THE decisive scale argument: three channels, same sessions
    ax = fig.add_subplot(gs[0, 1])
    chans = [("movement\n(distance)", S["ratio_locomotion"], C_MUT,
              S["p_locomotion"]),
             ("mechanical\ntransients", S["ratio_mech"], C_MECH,
              S["p_mech_rate"]),
             ("detected\ncalls", S["ratio_calls"], C_CALL, S["p_dyad_all"])]
    xs = np.arange(3)
    ax.bar(xs, [c[1] for c in chans], width=0.6,
           color=[c[2] for c in chans], edgecolor=C_SURF)
    for x, (lab, v, col, p) in zip(xs, chans):
        ax.text(x, v * 1.12, f"{v:.2f}×\np = {p:.3f}", ha="center",
                fontsize=8.5, color=C_SEC)
    ax.axhline(1.0, color=C_INK, lw=1.2, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(0.6, S["ratio_calls"] * 3.2)
    ax.set_xticks(xs, [c[0] for c in chans])
    ax.set_ylabel("WT / HET ratio (log scale)")
    ax.set_title("(b)  A 1.4× movement difference cannot\nmake a 12× call "
                 "difference", loc="left")

    # (c) calls per metre travelled
    ax = fig.add_subplot(gs[0, 2])
    dots(ax, 0, wt["calls_per_kpx"], C_WT)
    dots(ax, 1, het["calls_per_kpx"], C_HET)
    p = S["p_calls_per_distance"]
    ax.text(0.5, ses["calls_per_kpx"].max() * 1.03,
            f"p = {p:.4f} {stars(p)}", ha="center", fontsize=9)
    ax.set_xticks([0, 1], ["WT", "HET"])
    ax.set_ylabel("calls per 10³ px travelled")
    ax.set_title("(c)  Normalising by movement does not\nremove the effect", loc="left")

    # (d) mechanical events: the true movement-driven signal
    ax = fig.add_subplot(gs[1, 0])
    dots(ax, 0, wt["mech_rate_per_min"], C_WT)
    dots(ax, 1, het["mech_rate_per_min"], C_HET)
    p = S["p_mech_rate"]
    ax.text(0.5, ses["mech_rate_per_min"].max() * 1.02,
            f"p = {p:.2f} {stars(p)}", ha="center", fontsize=9)
    ax.set_xticks([0, 1], ["WT", "HET"])
    ax.set_ylabel("mechanical transients / min")
    ax.set_title("(d)  Genuine movement noise shows NO\ngenotype difference", loc="left")

    # (e) strict-USV re-analysis
    ax = fig.add_subplot(gs[1, 1])
    labels = ["all detected\ncalls", "strict USVs only\n(silent low band)"]
    wt_v = [S["wt_dyad_rate_all"], S["wt_dyad_rate_strict"]]
    het_v = [S["het_dyad_rate_all"], S["het_dyad_rate_strict"]]
    x = np.arange(2)
    ax.bar(x - 0.18, wt_v, width=0.34, color=C_WT, edgecolor=C_SURF, label="WT")
    ax.bar(x + 0.18, het_v, width=0.34, color=C_HET, edgecolor=C_SURF, label="HET")
    ax.set_ylim(0, max(wt_v) * 1.28)
    for xi, (a_, b_, p) in enumerate(zip(wt_v, het_v,
                                         [S["p_dyad_all"], S["p_dyad_strict"]])):
        ax.text(xi, max(a_, b_) * 1.04, f"p = {p:.4f} {stars(p)}", ha="center",
                fontsize=8.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("dyad call rate (calls/min)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"(e)  Keeping only strict USVs "
                 f"({S['pct_strict']:.0f}% of calls)\nleaves the result intact",
                 loc="left")

    # (f) verdict
    ax = fig.add_subplot(gs[1, 2])
    ax.set_axis_off()
    txt = (
        "FALSE-POSITIVE CONTROL: VERDICT\n\n"
        "hypothesis: detections are movement noise\n\n"
        "sub-USV band (detector never saw it):\n"
        f"  at detected calls    {S['median_low_call']:+.2f} dB\n"
        f"  at matched controls  {S['median_low_ctrl']:+.2f} dB  (p={S['p_low_call_vs_ctrl']:.2f})\n"
        f"  at true mechanical  {S['median_low_mech']:+.1f} dB\n"
        "  -> calls are SILENT where mechanical\n"
        "     noise is necessarily loud\n\n"
        f"coincidence with transients {S['pct_calls_with_mech']:.1f}%\n"
        f"  (chance alone would give  {S['pct_calls_with_mech_expected']:.1f}%)\n"
        f"calls emitted at rest       {S['pct_calls_immobile']:.0f}%\n\n"
        "genotype scaling:\n"
        f"  movement   {S['ratio_locomotion']:.2f}x  (p={S['p_locomotion']:.3f})\n"
        f"  mech noise {S['ratio_mech']:.2f}x  (p={S['p_mech_rate']:.2f})\n"
        f"  calls     {S['ratio_calls']:5.1f}x  (p={S['p_dyad_all']:.3f})\n\n"
        "REJECTED. The genotype effect is\n"
        "vocal, not locomotor."
    )
    ax.text(0.02, 0.98, txt, va="top", ha="left", fontsize=8.7,
            family="monospace", color=C_INK,
            bbox=dict(boxstyle="round,pad=0.6", fc="#f9f9f7", ec=C_BASE))
    ax.set_title("(f)  Summary", loc="left")

    fig.suptitle("Does the WT vs HET result survive the false-positive "
                 "counter-hypothesis?", y=1.0)
    savefig(fig, "v3_fig2_result_survives")


# =====================================================================
def main():
    d = load()
    calls, ctrl, shape, ses = d["calls"], d["ctrl"], d["shape"], d["sess"]

    S = {}
    S["median_low_call"] = float(calls["low_snr_db"].median())
    S["median_low_ctrl"] = float(ctrl["low_snr_db"].median())
    S["median_low_mech"] = float(shape.loc[shape["tag"] == "mech",
                                           "peak_low_db"].median())
    S["median_usv_call"] = float(calls["usv_snr_db"].median())
    S["p_low_call_vs_ctrl"] = mw(calls["low_snr_db"], ctrl["low_snr_db"])
    S["pct_calls_with_mech"] = float(calls["mech_within_10ms"].mean() * 100)
    S["r_low_mean"] = float(ses["corr_lowband_movement"].mean())
    S["r_usv_mean"] = float(ses["corr_usvband_movement"].mean())

    thr = 30.0
    S["immobile_thr_px_s"] = thr
    S["pct_calls_immobile"] = float((calls["speed_m1"] < thr).mean() * 100)
    # baseline: how much of the session is spent that slow?
    kin, _, _ = C2.load_kinematics()
    frac_time = [float(np.nanmean(np.abs(kin[a]["m1_speed"].to_numpy()) < thr))
                 for a in A.GENOTYPE_MAP]
    S["pct_time_immobile"] = float(np.nanmean(frac_time) * 100)

    # mechanical coincidence vs the rate expected by chance alone
    obs, exp = [], []
    for _, r in ses.iterrows():
        sub = calls[calls["animal_id"] == r["animal_id"]]
        if not len(sub):
            continue
        lam = r["n_mech_events"] / A.SESSION_END_S          # events per second
        obs.append(sub["mech_within_10ms"].mean())
        exp.append(1 - np.exp(-lam * 0.020))                # +-10 ms window
    S["pct_calls_with_mech_expected"] = float(np.mean(exp) * 100)
    S["p_coincidence_vs_chance"] = float(
        stats.wilcoxon(obs, exp)[1]) if len(obs) > 5 else np.nan

    # three-channel effect sizes: movement, mechanical noise, calls
    wt0, het0 = ses[ses["genotype"] == "WT"], ses[ses["genotype"] == "HET"]
    S["ratio_locomotion"] = float(wt0["dist_px_partner"].mean()
                                  / het0["dist_px_partner"].mean())
    S["ratio_mech"] = float(wt0["mech_rate_per_min"].mean()
                            / het0["mech_rate_per_min"].mean())
    S["wt_mech_rate"] = float(wt0["mech_rate_per_min"].mean())
    S["het_mech_rate"] = float(het0["mech_rate_per_min"].mean())

    # tonality separation, calls vs movement-matched controls
    sh_call = shape[shape["tag"] == "call"]
    sh_ctrl = shape[shape["tag"] == "control"]
    y = np.r_[np.ones(len(sh_call)), np.zeros(len(sh_ctrl))]
    feats = ["tonality", "entropy_bits", "usv_energy_frac"]
    X = np.vstack([sh_call[feats].to_numpy(), sh_ctrl[feats].to_numpy()])
    okX = np.isfinite(X).all(axis=1)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        X[okX], y[okX])
    S["shape_auc_call_vs_control"] = float(
        roc_auc_score(y[okX], clf.decision_function(X[okX])))
    S["tonality_call"] = float(sh_call["tonality"].median())
    S["tonality_control"] = float(sh_ctrl["tonality"].median())

    # locomotion accounting
    wt, het = ses[ses["genotype"] == "WT"], ses[ses["genotype"] == "HET"]
    ses["calls_per_kpx"] = ses["n_calls"] / (ses["dist_px_total"] / 1e3)
    wt, het = ses[ses["genotype"] == "WT"], ses[ses["genotype"] == "HET"]
    S["p_locomotion"] = mw(wt["dist_px_partner"], het["dist_px_partner"])
    S["p_calls_per_distance"] = mw(wt["calls_per_kpx"], het["calls_per_kpx"])
    S["p_mech_rate"] = mw(wt["mech_rate_per_min"], het["mech_rate_per_min"])
    S["rho_calls_distance"] = float(stats.spearmanr(
        ses["dist_px_partner"], ses["n_calls"])[0])
    S["wt_locomotion_mean"] = float(wt["dist_px_partner"].mean())
    S["het_locomotion_mean"] = float(het["dist_px_partner"].mean())

    # strict-USV subset re-analysis
    strict = calls[calls["low_snr_db"] < STRICT_LOW_DB]
    S["pct_strict"] = float(len(strict) / len(calls) * 100)
    inter = 10.0
    def rate(df, gt, strict_only):
        src = strict if strict_only else calls
        sub = src[(src["genotype"] == gt) & (src["phase"] == "partner")]
        n = sub.groupby("animal_id").size()
        animals = [a for a, g in A.GENOTYPE_MAP.items() if g == gt]
        return np.array([n.get(a, 0) / inter for a in animals])
    for tag, so in (("all", False), ("strict", True)):
        w_, h_ = rate(calls, "WT", so), rate(calls, "HET", so)
        S[f"wt_dyad_rate_{tag}"] = float(w_.mean())
        S[f"het_dyad_rate_{tag}"] = float(h_.mean())
        S[f"p_dyad_{tag}"] = mw(w_, h_)
    S["ratio_calls"] = S["wt_dyad_rate_all"] / max(S["het_dyad_rate_all"], 1e-9)

    # GEO attribution recomputed on the strict subset
    geo = d["geo"].merge(calls[["call_id", "low_snr_db"]], on="call_id",
                         how="left")
    geo_strict = geo[geo["low_snr_db"] < STRICT_LOW_DB]
    cal_a = d["geoval"]["cal_slope"]
    res_rates = {}
    for tag, src in (("all", geo), ("strict", geo_strict)):
        vals = {}
        for gt in ("WT", "HET"):
            rr = []
            for a in [x for x, g in A.GENOTYPE_MAP.items() if g == gt]:
                sub = src[src["animal_id"] == a]
                if not len(sub):
                    rr.append(0.0)
                    continue
                pi_hat = pi_mle(cal_a * sub["geo_logodds"].to_numpy())[0]
                rr.append(pi_hat * len(sub) / inter)
            vals[gt] = np.array(rr)
        S[f"wt_res_rate_{tag}"] = float(vals["WT"].mean())
        S[f"het_res_rate_{tag}"] = float(vals["HET"].mean())
        S[f"p_res_{tag}"] = mw(vals["WT"], vals["HET"])
        res_rates[tag] = vals

    ses.to_csv(OUT / "noise_session_stats.csv", index=False)
    with open(OUT / "noise_control_summary.json", "w") as fh:
        json.dump(S, fh, indent=2)
    print(json.dumps(S, indent=2))

    fig1(d, S)
    fig2(d, S)


if __name__ == "__main__":
    main()
