"""Figures for the single-mic call-attribution project.

Fig 1  METHOD        pipeline, example calls, from-scratch AE, voice space,
                     intensity-cue test
Fig 2  VALIDATION    ground-truth benchmarks: the per-call attribution verdict
Fig 3  QUANTIFICATION design-based bounds: WT vs HET from the resident's and
                     the partner's point of view

Outputs attribution/attr_fig{1,2,3}_*.png/.svg/.pdf
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A
import attr_core as CORE

# validated palette (dataviz reference, light mode)
C_WT = "#2a78d6"
C_HET = "#eb6834"
C_RES = "#1baf7a"
C_PART = "#4a3aa7"
C_INK = "#0b0b0b"
C_SEC = "#52514e"
C_MUT = "#898781"
C_GRID = "#e1e0d9"
C_SURF = "#fcfcfb"
C_BASE = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": C_SURF, "axes.facecolor": C_SURF,
    "savefig.facecolor": C_SURF, "axes.edgecolor": C_BASE,
    "axes.labelcolor": C_SEC, "text.color": C_INK,
    "xtick.color": C_MUT, "ytick.color": C_MUT,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "figure.titlesize": 14, "figure.titleweight": "bold",
    "svg.fonttype": "none",
})

OUT = A.ensure_out()


def savefig(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def load_everything():
    feats = pd.read_csv(OUT / "call_features.csv")
    feats["animal_id"] = feats["animal_id"].astype(str)
    feats = A.apply_genotype(feats)
    emb = np.load(OUT / "call_embeddings.npz", allow_pickle=False)
    lat = pd.DataFrame(emb["latent"],
                       columns=[f"z{i}" for i in range(emb["latent"].shape[1])])
    lat["call_id"] = emb["call_id"]
    df = feats.merge(lat, on="call_id").reset_index(drop=True)
    zcols = [c for c in df.columns if c.startswith("z") and c[1:].isdigit()]
    specs = np.load(OUT / "call_specs.npz", allow_pickle=False)
    att = A.apply_genotype(pd.read_csv(OUT / "call_attribution.csv")
                           .assign(animal_id=lambda d: d["animal_id"].astype(str)))
    ses = pd.read_csv(OUT / "session_quantification.csv")
    ses["animal_id"] = ses["animal_id"].astype(str)
    bench_pairs = pd.read_csv(OUT / "benchmark_pairs.csv")
    nc = pd.read_csv(OUT / "benchmark_negative_control.csv")
    with open(OUT / "benchmark_summary.json") as fh:
        bench = json.load(fh)
    with open(OUT / "quantification_summary.json") as fh:
        quant = json.load(fh)
    model = np.load(OUT / "attribution_model.npz", allow_pickle=False)
    return df, zcols, emb, specs, att, ses, bench_pairs, nc, bench, quant, model


# ===========================================================================
def fig1_method(df, zcols, emb, specs, model):
    fig = plt.figure(figsize=(13.5, 9.5))
    gs = fig.add_gridspec(3, 4, hspace=0.52, wspace=0.34,
                          height_ratios=[1.0, 1.05, 1.15])

    # (a) pipeline schematic ------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    boxes = [
        (1.0, "single mic 384 kHz\n+ video tracking\n(24 sessions)", C_MUT),
        (18.5, f"{len(df):,} detected calls\nridge features (18)\n64x64 SNR crops", C_WT),
        (36.0, "from-scratch conv-AE\n(this data only,\nno checkpoint)", C_WT),
        (53.5, "voice space (PCA-12)\nanchored 2-source EM\n+ bout HMM + level cue", C_RES),
        (71.0, "GROUND-TRUTH\nBENCHMARKS\npseudo/audio dyads", C_HET),
        (88.0, "verdict gates use:\nper-call -> rejected\ndesign bounds -> used", C_INK),
    ]
    for x, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 2.2), 11.5, 6.2,
                                    boxstyle="round,pad=0.35",
                                    fc=C_SURF, ec=col, lw=1.6))
        ax.text(x + 5.75, 5.3, txt, ha="center", va="center", fontsize=8,
                color=C_INK)
    for x in (13.0, 30.5, 48.0, 65.5, 83.0):
        ax.add_patch(FancyArrowPatch((x, 5.3), (x + 5.0, 5.3),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=C_MUT, lw=1.4))
    ax.set_title("(a)  Attribution pipeline: every component is benchmarked before its output is trusted",
                 loc="left")

    # (b) example call crops -----------------------------------------------
    ids = specs["call_id"]
    S = specs["specs"].astype(float)
    id2row = {c: i for i, c in enumerate(ids)}
    rng = np.random.default_rng(3)
    axb = fig.add_subplot(gs[1, :2])
    axb.set_axis_off()
    picks = {}
    for phase in ("alone", "partner"):
        cand = df[(df["phase"] == phase) & (df["rf_dur_ms"] > 5)]
        picks[phase] = rng.choice(cand["call_id"].to_numpy(), 6, replace=False)
    canvas = np.ones((2 * 66, 6 * 66)) * np.nan
    for r, phase in enumerate(("alone", "partner")):
        for k, cid in enumerate(picks[phase]):
            canvas[r * 66 + 1:r * 66 + 65, k * 66 + 1:k * 66 + 65] = \
                S[id2row[cid]][::-1]
    axb.imshow(canvas, cmap="magma", aspect="auto", interpolation="nearest")
    axb.text(-6, 33, "alone\n(resident\ncertain)", ha="right", va="center",
             fontsize=8, color=C_RES)
    axb.text(-6, 99, "partner\nphase\n(caller ?)", ha="right", va="center",
             fontsize=8, color=C_PART)
    axb.set_title("(b)  Example call crops (40-125 kHz x call span)", loc="left")

    # (c) AE reconstructions + loss ------------------------------------------
    axc = fig.add_subplot(gs[1, 2])
    axc.set_axis_off()
    orig = emb["example_orig"].astype(float)
    rec = emb["example_rec"].astype(float)
    canv = np.ones((2 * 66, 4 * 66)) * np.nan
    for k in range(4):
        canv[1:65, k * 66 + 1:k * 66 + 65] = orig[k][::-1]
        canv[67:131, k * 66 + 1:k * 66 + 65] = rec[k][::-1]
    axc.imshow(canv, cmap="magma", aspect="auto", interpolation="nearest")
    axc.text(-6, 33, "input", ha="right", va="center", fontsize=8, color=C_SEC)
    axc.text(-6, 99, "recon", ha="right", va="center", fontsize=8, color=C_SEC)
    axc.set_title("(c)  From-scratch AE", loc="left")

    axl = fig.add_subplot(gs[1, 3])
    loss = emb["loss_curve"]
    axl.plot(np.arange(1, len(loss) + 1), loss, color=C_WT, lw=2)
    axl.set_xlabel("epoch")
    axl.set_ylabel("reconstruction MSE")
    axl.set_title("(d)  AE training", loc="left")
    axl.annotate(f"{loss[-1]:.4f}", (len(loss), loss[-1]),
                 textcoords="offset points", xytext=(-8, 8),
                 color=C_WT, fontsize=8, ha="right")

    # (e) voice space: context shift -----------------------------------------
    axe = fig.add_subplot(gs[2, :2])
    X, _, _ = CORE.build_voice_space(df, df[zcols].to_numpy())
    al = (df["phase"] == "alone").to_numpy()
    axe.scatter(X[~al, 0], X[~al, 1], s=9, c=C_PART, alpha=0.25,
                edgecolors="none", label="partner phase (caller unknown)")
    axe.scatter(X[al, 0], X[al, 1], s=26, c=C_RES, alpha=0.9,
                edgecolors="white", linewidths=0.4,
                label="alone phase (resident certain)")
    for m, lab, col in ((al, "alone", C_RES), (~al, "partner", C_PART)):
        mu = X[m, :2].mean(axis=0)
        axe.scatter(*mu, s=160, marker="X", c=col, edgecolors="white",
                    linewidths=1.2, zorder=5)
    axe.set_xlabel("voice PC1")
    axe.set_ylabel("voice PC2")
    axe.legend(loc="upper right", fontsize=8, frameon=False)
    axe.set_title("(e)  Voice space: alone->social repertoire shift\n"
                  "(tonality d=+2.1, duration d=+0.9) dominates PC1", loc="left")

    # (f) intensity-distance cue fails ---------------------------------------
    axf = fig.add_subplot(gs[2, 2:])
    alone_df = df[al]
    mic = model["mic"]
    d = np.hypot(alone_df["m1_x"] - mic[0], alone_df["m1_y"] - mic[1])
    L = alone_df["rf_amp_peak_db"]
    ok = np.isfinite(d) & np.isfinite(L)
    axf.scatter(d[ok], L[ok], s=18, c=C_RES, alpha=0.7, edgecolors="none")
    dd = np.linspace(max(d[ok].min(), 20), d[ok].max(), 100)
    pred = model["a0"] - 10 * model["b"] * np.log10(dd)
    axf.plot(dd, pred, color=C_HET, lw=2, ls="--",
             label=f"best fit (b={float(model['b']):.2f})")
    axf.set_xlabel("caller distance to fitted mic position (px)")
    axf.set_ylabel("received peak level (dB SNR)")
    axf.legend(loc="upper right", fontsize=8, frameon=False)
    axf.set_title(f"(f)  No usable level-distance law:\nleave-one-animal-out "
                  f"R² = {float(model['r2_loao']):.2f} -> cue disabled", loc="left")

    fig.suptitle("Single-microphone USV attribution - method (LgDel WT vs HET, "
                 "24 residents, 17 identified stim partners)", y=0.995)
    savefig(fig, "attr_fig1_method")


# ===========================================================================
def fig2_validation(df, zcols, bench_pairs, nc, bench):
    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.32)

    # (a) identity signal across representations ---------------------------
    ax = fig.add_subplot(gs[0, 0])
    alone = df[df["phase"] == "alone"]
    ids = alone["animal_id"].to_numpy()

    def prep(M):
        M = np.where(np.isfinite(M), M, np.nan)
        med = np.nanmedian(M, axis=0)
        return StandardScaler().fit_transform(np.where(np.isnan(M), med, M))

    H = prep(alone[CORE.VOICE_FEATS].to_numpy(float))
    Z = prep(alone[zcols].to_numpy(float))
    reps = {"hand-\ncrafted": H, "AE\nlatent": Z,
            "combined": np.hstack([H, Z]),
            "freq\nonly": prep(alone[["rf_f_mean_khz", "rf_f_min_khz",
                                      "rf_f_max_khz", "rf_f_start_khz",
                                      "rf_f_end_khz"]].to_numpy(float))}
    aucs = []
    for name, Xr in reps.items():
        d2 = np.sum((Xr[:, None] - Xr[None, :]) ** 2, axis=2)
        iu = np.triu_indices(len(Xr), k=1)
        same = (ids[iu[0]] == ids[iu[1]]).astype(int)
        aucs.append(roc_auc_score(same, -d2[iu]))
    xs = np.arange(len(reps))
    ax.bar(xs, aucs, width=0.62, color=C_WT, edgecolor=C_SURF)
    for x, v in zip(xs, aucs):
        ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=8, color=C_SEC)
    ax.axhline(0.5, color=C_INK, lw=1.2, ls="--")
    ax.text(len(reps) - 0.4, 0.505, "chance", fontsize=8, color=C_INK,
            ha="right", va="bottom")
    ax.set_xticks(xs, list(reps))
    ax.set_ylim(0.4, 0.75)
    ax.set_ylabel("same vs different animal AUC")
    ax.set_title("(a)  Individual voice fingerprint:\nabsent in every representation",
                 loc="left")

    # (b) speaker ID vs permutation null -----------------------------------
    ax = fig.add_subplot(gs[0, 1])
    obs = bench["speakerid_acc"] * 100
    chance = bench["speakerid_chance"] * 100
    nullm = bench["speakerid_null_mean"] * 100
    bars = ax.bar([0, 1, 2], [obs, nullm, chance], width=0.62,
                  color=[C_WT, C_MUT, C_BASE], edgecolor=C_SURF)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8, color=C_SEC)
    ax.set_xticks([0, 1, 2], ["observed", "shuffled\nlabels", "chance"])
    ax.set_ylabel("17-way speaker-ID accuracy (%)")
    ax.set_title(f"(b)  Speaker ID on alone calls:\n"
                 f"n.s. vs null (perm p = {bench['speakerid_perm_p']:.2f})",
                 loc="left")

    # (c) pseudo-dyad AUCs --------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(0)
    for i, (tag, col, label) in enumerate(
            [("feature", C_WT, "feature-level\n(198 pairs)"),
             ("audio", C_HET, "audio mixtures\n(12 pairs)")]):
        v = bench_pairs.loc[bench_pairs["tag"] == tag, "auc_hmm"].to_numpy()
        jit = rng.normal(0, 0.06, len(v))
        ax.scatter(np.full(len(v), i) + jit, v, s=14, c=col, alpha=0.55,
                   edgecolors="none")
        ax.hlines(np.mean(v), i - 0.22, i + 0.22, color=col, lw=3)
        ax.text(i, np.mean(v) + 0.03, f"mean {np.mean(v):.2f}", ha="center",
                fontsize=8, color=col)
    ax.axhline(0.5, color=C_INK, lw=1.2, ls="--")
    ax.set_xticks([0, 1], ["feature-level\n(198 pairs)", "audio mixtures\n(12 pairs)"])
    ax.set_ylabel("attribution AUC vs ground truth")
    ax.set_ylim(0, 1.05)
    ax.set_title("(c)  Ground-truth synthetic dyads:\nattribution at chance",
                 loc="left")

    # (d) negative control --------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(nc["p_res"], bins=np.linspace(0, 1, 21), color=C_RES,
            edgecolor=C_SURF)
    frac = (nc["p_res"] > 0.5).mean() * 100
    ax.axvline(0.5, color=C_INK, lw=1.2, ls="--")
    ax.set_xlabel("P(resident) assigned to held-out ALONE calls")
    ax.set_ylabel("calls")
    ax.set_title(f"(d)  Negative control: known-resident calls\n"
                 f"recovered only {frac:.0f}% (should be ~100%)", loc="left")

    # (e) identity vs context shift per feature ----------------------------
    ax = fig.add_subplot(gs[1, 1])
    from scipy import stats as st
    part = df[df["phase"] == "partner"]
    big = [a for a, n in alone.groupby("animal_id").size().items() if n >= 4]
    pts = []
    for c in CORE.VOICE_FEATS:
        groups = [g[c].dropna().to_numpy()
                  for _, g in alone[alone["animal_id"].isin(big)].groupby("animal_id")]
        groups = [g for g in groups if len(g) >= 3]
        F = st.f_oneway(*groups)[0] if len(groups) > 2 else np.nan
        alx, pax = alone[c].dropna(), part[c].dropna()
        dsh = abs(pax.mean() - alx.mean()) / np.sqrt((alx.var() + pax.var()) / 2)
        pts.append((dsh, F, c))
    dsh, Fv, names = zip(*pts)
    ax.scatter(dsh, Fv, s=26, c=C_PART, alpha=0.8, edgecolors="white",
               linewidths=0.4)
    for x, y, nm in pts:
        if x > 0.8 or y > 2.0:
            ax.annotate(nm.replace("rf_", ""), (x, y),
                        textcoords="offset points", xytext=(4, 3), fontsize=7,
                        color=C_SEC)
    ax.axhline(1.9, color=C_MUT, lw=1, ls=":")
    ax.text(ax.get_xlim()[1], 1.95, "identity signal p<0.05", fontsize=7,
            color=C_MUT, ha="right", va="bottom")
    ax.set_xlabel("context shift |d| (alone -> partner phase)")
    ax.set_ylabel("individual identity F (alone calls)")
    ax.set_title("(e)  Context shift dwarfs identity:\nmixture splits call types, "
                 "not callers", loc="left")

    # (f) verdict text ------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.set_axis_off()
    txt = (
        "VERDICT (all computed on this dataset)\n\n"
        "REJECTED - per-call attribution:\n"
        f"  voice fingerprint AUC {bench['auc_same_vs_diff']:.2f} (chance)\n"
        f"  pseudo-dyad AUC {bench['feature_mean_auc_hmm']:.2f}"
        f" / audio {bench['audio_mean_auc_hmm']:.2f}\n"
        f"  negative control {bench['negative_control_frac_res']*100:.0f}%"
        f" (need ~100%)\n"
        f"  level-distance law R² ≈ 0\n\n"
        "SUPPORTED - design-based attribution:\n"
        "  alone phase = pure resident (by design)\n"
        "  HET dyads cap the partner's contribution\n"
        "  (stim pool counterbalanced, 51557 met\n"
        "   both genotypes)\n"
        "  shared-stim natural experiment: same\n"
        "  partner, 38x different output (51558)"
    )
    ax.text(0.02, 0.98, txt, va="top", ha="left", fontsize=8.6, color=C_INK,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", fc="#f9f9f7", ec=C_BASE))
    ax.set_title("(f)  What the benchmarks allow", loc="left")

    fig.suptitle("Validation: can single-mic calls be attributed per call? "
                 "The benchmarks say no - so we use the design instead", y=0.995)
    savefig(fig, "attr_fig2_validation")


# ===========================================================================
def fig3_quantification(att, ses, quant):
    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.36)
    wt = ses[ses["genotype"] == "WT"]
    het = ses[ses["genotype"] == "HET"]
    ceiling = quant["partner_ceiling_calls_per_min"]

    def dots(ax, x, vals, color):
        rng = np.random.default_rng(1)
        jit = rng.normal(0, 0.05, len(vals))
        ax.scatter(np.full(len(vals), x) + jit, vals, s=26, c=color, alpha=0.75,
                   edgecolors="white", linewidths=0.4, zorder=3)
        ax.hlines(np.median(vals), x - 0.22, x + 0.22, color=color, lw=3, zorder=4)

    # (a) rates alone vs interaction ---------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for i, (col_, g) in enumerate((("alone_rate", wt), ("alone_rate", het),
                                   ("dyad_rate", wt), ("dyad_rate", het))):
        col = C_WT if g is wt else C_HET
        dots(ax, [0, 0.7, 2.0, 2.7][i], g[col_], col)
    ax.set_xticks([0.35, 2.35], ["alone\n(0-5 min)", "interaction\n(5-15 min)"])
    ax.set_ylabel("dyad calls / min")
    ax.text(0.02, 0.97, "WT", transform=ax.transAxes, color=C_WT,
            fontweight="bold")
    ax.text(0.14, 0.97, "HET", transform=ax.transAxes, color=C_HET,
            fontweight="bold")
    stats_df = pd.read_csv(OUT / "attribution_statistics.csv").set_index("metric")
    p_dyad = stats_df.loc["dyad_rate", "mannwhitney_p"]
    ax.text(2.35, max(wt["dyad_rate"]) * 1.02,
            f"p = {p_dyad:.3f} {stats_df.loc['dyad_rate', 'sig']}", ha="center",
            fontsize=8.5, color=C_INK)
    ax.text(0.35, max(wt["alone_rate"]) + 1.2, "ns", ha="center", fontsize=8.5,
            color=C_MUT)
    ax.set_title("(a)  Total (unattributed) call rates", loc="left")

    # (b) time course -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1:])
    tc = pd.read_csv(OUT / "timecourse_30s.csv")
    tc["animal_id"] = tc["animal_id"].astype(str)
    bins = np.arange(0, 900, 30)
    stats_df = pd.read_csv(OUT / "attribution_statistics.csv").set_index("metric")
    for gt, col in (("WT", C_WT), ("HET", C_HET)):
        animals = [a for a, g in A.GENOTYPE_MAP.items() if g == gt]
        M = np.zeros((len(animals), len(bins)))
        for i, a in enumerate(animals):
            sub = tc[tc["animal_id"] == a].set_index("bin_s")["n_calls"]
            M[i] = [sub.get(b, 0) * 2 for b in bins]      # calls/min
        mean = M.mean(axis=0)
        sem = M.std(axis=0, ddof=1) / np.sqrt(len(animals))
        ax.plot(bins / 60, mean, color=col, lw=2, label=gt)
        ax.fill_between(bins / 60, mean - sem, mean + sem, color=col, alpha=0.18,
                        edgecolor="none")
        peak = int(np.argmax(mean))
        dy = 10 if gt == "WT" else -12
        ax.annotate(gt, (bins[peak] / 60, mean[peak]),
                    textcoords="offset points", xytext=(10, dy), color=col,
                    fontweight="bold", fontsize=9)
    ax.axvline(5, color=C_INK, lw=1.2, ls="--")
    ax.text(5.08, ax.get_ylim()[1] * 0.95, "partner in", fontsize=8.5,
            color=C_INK, va="top")
    ax.set_xlabel("session time (min)")
    ax.set_ylabel("dyad calls / min (mean ± SEM)")
    ax.set_title("(b)  Calling explodes at partner introduction in WT dyads only",
                 loc="left")

    # (c) shared-stim natural experiment -----------------------------------
    ax = fig.add_subplot(gs[1, 0])
    sh = pd.read_csv(OUT / "shared_stim_experiment.csv")
    sh = sh[sh["n_sessions_for_stim"] >= 2].copy()
    sh["animal_id"] = sh["animal_id"].astype(str)
    xpos, labels = [], []
    x = 0.0
    for stim, g in sh.groupby("stim_id"):
        for _, r in g.iterrows():
            col = C_WT if r["genotype"] == "WT" else C_HET
            ax.bar(x, max(r["n_dyad_calls"], 0.6), width=0.62, color=col,
                   edgecolor=C_SURF)
            ax.text(x, max(r["n_dyad_calls"], 0.6) * 1.08,
                    f"{int(r['n_dyad_calls'])}", ha="center", fontsize=8.5,
                    color=C_SEC)
            labels.append(f"res {r['animal_id']}\n({r['genotype']})")
            xpos.append(x)
            x += 0.8
        x += 0.7
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 4)
    ax.set_xticks(xpos, labels, fontsize=7.5)
    ax.set_ylabel("dyad calls in interaction window (log)")
    ax.text(0.25, 1.005, "same partner: 51557", transform=ax.transAxes,
            ha="center", fontsize=8, color=C_SEC)
    ax.text(0.78, 1.005, "same partner: 51558", transform=ax.transAxes,
            ha="center", fontsize=8, color=C_SEC)
    ax.set_title("(c)  Same partner, different resident:\noutput tracks the resident\n",
                 loc="left")

    # (d) resident perspective ---------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    for i, (g, col, gt) in enumerate(((wt, C_WT, "WT"), (het, C_HET, "HET"))):
        s = g.sort_values("dyad_rate", ascending=False).reset_index(drop=True)
        xs = np.arange(len(s)) + i * (len(wt) + 2)
        ax.bar(xs, s["res_rate_upper"] - s["res_rate_lower"],
               bottom=s["res_rate_lower"], width=0.7, color=col, alpha=0.35,
               edgecolor="none")
        ax.scatter(xs, (s["res_rate_lower"] + s["res_rate_upper"]) / 2, s=14,
                   c=col, edgecolors="none")
        ax.text(xs.mean(), ax.get_ylim()[1] * 0.0 + max(wt["dyad_rate"]) * 0.97,
                gt, ha="center", color=col, fontweight="bold")
    ax.axhline(ceiling, color=C_INK, lw=1, ls=":")
    ax.text(0, ceiling + 0.6, f"partner ceiling {ceiling:.1f}/min", fontsize=7.5,
            color=C_INK)
    ax.set_xticks([])
    ax.set_xlabel("sessions (sorted)")
    ax.set_ylabel("RESIDENT-attributed calls/min\n[bound interval]")
    p_res = stats_df.loc["res_rate_lower", "mannwhitney_p"]
    ax.set_title("(d)  Experimental mouse's view:\nWT lower bound "
                 f"{wt['res_rate_lower'].mean():.1f} vs HET "
                 f"{het['res_rate_lower'].mean():.2f} (p={p_res:.4f})", loc="left")

    # (e) partner perspective ----------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    for i, (g, col, gt) in enumerate(((wt, C_WT, "with WT\nresident"),
                                      (het, C_HET, "with HET\nresident"))):
        dots(ax, i, g["part_rate_upper"], col)
    ax.axhline(ceiling, color=C_INK, lw=1, ls=":")
    ax.text(0.5, ceiling * 1.06, f"partner ceiling {ceiling:.1f}/min", fontsize=7.5,
            color=C_INK, ha="center")
    ax.set_ylim(top=ceiling * 1.18)
    ax.set_xticks([0, 1], ["with WT\nresident", "with HET\nresident"])
    ax.set_ylabel("PARTNER-attributed calls/min\n(UPPER bounds, not point estimates)")
    ax.set_title("(e)  Partner's view: contribution is small\nregardless of resident "
                 f"genotype (≤{ceiling:.1f}/min)", loc="left")

    share = quant["wt_resident_share_lower"] * 100
    fig.suptitle(
        f"Quantification with design-based attribution: the interaction-window surge "
        f"belongs to the WT resident (≥{share:.0f}% of WT dyad calls)", y=1.0)
    savefig(fig, "attr_fig3_quantification")


def main():
    df, zcols, emb, specs, att, ses, bench_pairs, nc, bench, quant, model = \
        load_everything()
    fig1_method(df, zcols, emb, specs, model)
    fig2_validation(df, zcols, bench_pairs, nc, bench)
    fig3_quantification(att, ses, quant)


if __name__ == "__main__":
    main()
