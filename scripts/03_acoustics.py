"""MODULE C - Acoustic properties WT vs Het (session 6).

Four acoustic descriptors, at the individual-call level and across phases:
  * dur_ms      call duration (ms)
  * avg_khz     mean frequency (kHz)
  * bw_khz      bandwidth (kHz)  = tonal spread
  * contrast_db avg_intensity - bg_intensity  (tonal contrast vs background)

Outputs (via C.save_fig):
  fig03_acoustics_violin.png   4-panel call-level violin+box, WT vs Het + MW brackets
  fig03_by_phase.png           4 metrics as mean +/- sem across phases, one line/geno
  fig03_dur_freq_scatter.png   duration(log) vs mean-freq scatter + marginal hists

Stats: for each metric we report WT/Het medians, a CALL-LEVEL Mann-Whitney
(U, p, rank-biserial), and an ANIMAL-LEVEL Mann-Whitney on per-mouse medians
(n_WT=2 mice vs n_Het=4 mice -> underpowered, reported honestly).
"""
import sys
sys.path.insert(0, "/home/andry/UVS/scripts")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import common as C

C.apply_style()

METRICS = [
    ("dur_ms", "duration (ms)", True),      # log-x candidate / heavy-tailed
    ("avg_khz", "mean frequency (kHz)", False),
    ("bw_khz", "bandwidth (kHz)", False),
    ("contrast_db", "tonal contrast (dB)", False),
]


def load():
    df = C.load_calls()
    df["contrast_db"] = df["avg_intensity"] - df["bg_intensity"]
    return df


def print_stats(df):
    """Call-level + animal-level MW for each metric; returns dict of results."""
    res = {}
    print("\n" + "=" * 78)
    print("MODULE C  -  ACOUSTIC PROPERTIES  WT (n=%d calls / 2 mice)  vs  Het (n=%d calls / 4 mice)"
          % ((df.geno == "WT").sum(), (df.geno == "Het").sum()))
    print("=" * 78)
    for key, label, _ in METRICS:
        wt = df.loc[df.geno == "WT", key].to_numpy(float)
        he = df.loc[df.geno == "Het", key].to_numpy(float)
        med_wt, med_he = np.nanmedian(wt), np.nanmedian(he)
        U, p, rbc = C.mannwhitney(wt, he)
        # animal-level: per-mouse medians
        pm = df.groupby(["geno", "mouse"], observed=True)[key].median().reset_index()
        wt_m = pm.loc[pm.geno == "WT", key].to_numpy(float)
        he_m = pm.loc[pm.geno == "Het", key].to_numpy(float)
        Ua, pa, rbca = C.mannwhitney(wt_m, he_m)
        res[key] = dict(med_wt=med_wt, med_he=med_he, U=U, p=p, rbc=rbc,
                        p_animal=pa, rbc_animal=rbca, wt_m=wt_m, he_m=he_m)
        print(f"\n{label} [{key}]")
        print(f"  median   WT = {med_wt:9.4f}   Het = {med_he:9.4f}   (WT/Het = {med_wt/med_he:5.2f}x)"
              if med_he else f"  median   WT = {med_wt:9.4f}   Het = {med_he:9.4f}")
        print(f"  call-level  U={U:.0f}  p={p:.3e} {C.pstars(p):>4}  rank-biserial={rbc:+.3f}")
        print(f"  per-mouse medians  WT={np.round(wt_m,3).tolist()}  Het={np.round(he_m,3).tolist()}")
        print(f"  animal-level MW    p={pa:.3e} {C.pstars(pa):>4}  rank-biserial={rbca:+.3f}   (n=2 vs 4, underpowered)")
    return res


def fig_violin(df, res):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    order = ["WT", "Het"]
    for ax, (key, label, _) in zip(axes, METRICS):
        data = [df.loc[df.geno == g, key].dropna().to_numpy(float) for g in order]
        parts = ax.violinplot(data, positions=[0, 1], widths=0.8,
                              showextrema=False)
        for pc, g in zip(parts["bodies"], order):
            pc.set_facecolor(C.GENO_COLOR[g]); pc.set_alpha(0.35)
            pc.set_edgecolor(C.GENO_COLOR[g]); pc.set_linewidth(1.0)
        bp = ax.boxplot(data, positions=[0, 1], widths=0.22, patch_artist=True,
                        showfliers=False, medianprops=dict(color="#111111", lw=1.6),
                        whiskerprops=dict(color="#555555"), capprops=dict(color="#555555"))
        for patch, g in zip(bp["boxes"], order):
            patch.set_facecolor(C.GENO_COLOR[g]); patch.set_alpha(0.8)
            patch.set_edgecolor("#333333")
        # jittered raw points
        for xpos, g in zip([0, 1], order):
            y = df.loc[df.geno == g, key].dropna().to_numpy(float)
            x = xpos + (np.random.RandomState(0).rand(len(y)) - 0.5) * 0.18
            ax.scatter(x, y, s=6, color=C.GENO_COLOR[g], alpha=0.35,
                       edgecolors="none", zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(order)
        ax.set_title(label)
        if key == "dur_ms":
            ax.set_yscale("log")
        # significance bracket (call-level)
        top = max(np.nanpercentile(data[0], 99), np.nanpercentile(data[1], 99))
        if key == "dur_ms":
            ax.set_ylim(top=top * 3)
            ybr = top * 1.5
        else:
            lo = min(np.nanmin(data[0]), np.nanmin(data[1]))
            ax.set_ylim(top=top + (top - lo) * 0.28)
            ybr = top + (top - lo) * 0.10
        C.sig_bracket(ax, 0, 1, ybr, res[key]["p"])
    axes[0].legend(handles=[Patch(facecolor=C.GENO_COLOR["WT"], label="WT (2 mice)"),
                            Patch(facecolor=C.GENO_COLOR["Het"], label="Het (4 mice)")],
                   loc="upper left", fontsize=9)
    fig.suptitle("Call-level acoustic properties  (session 6, all phases pooled)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    C.save_fig(fig, "fig03_acoustics_violin.png")
    plt.close(fig)


def fig_by_phase(df):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    xph = np.arange(len(C.PHASE_ORDER))
    for ax, (key, label, _) in zip(axes, METRICS):
        for g in ["WT", "Het"]:
            sub = df[df.geno == g]
            means, sems, ns = [], [], []
            for ph in C.PHASE_ORDER:
                v = sub.loc[sub.phase == ph, key].dropna().to_numpy(float)
                ns.append(len(v))
                means.append(np.mean(v) if len(v) else np.nan)
                sems.append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
            means, sems = np.array(means), np.array(sems)
            ax.errorbar(xph, means, yerr=sems, marker="o", ms=6, lw=2,
                        capsize=3, color=C.GENO_COLOR[g], label=g)
            for x, m, n in zip(xph, means, ns):
                if np.isfinite(m):
                    ax.annotate(f"n={n}", (x, m), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=7,
                                color=C.GENO_COLOR[g])
        ax.set_xticks(xph)
        ax.set_xticklabels([C.PHASE_LABEL[p] for p in C.PHASE_ORDER])
        ax.set_title(label)
    axes[0].set_ylabel("mean +/- sem")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Acoustic properties across social phases  (mean +/- sem; small n annotated)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    C.save_fig(fig, "fig03_by_phase.png")
    plt.close(fig)


def fig_scatter(df):
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    axtop = fig.add_subplot(gs[0, 0], sharex=ax)
    axright = fig.add_subplot(gs[1, 1], sharey=ax)
    for g in ["WT", "Het"]:
        sub = df[df.geno == g]
        d = sub["dur_ms"].to_numpy(float)
        f = sub["avg_khz"].to_numpy(float)
        ax.scatter(d, f, s=14, color=C.GENO_COLOR[g], alpha=0.45,
                   edgecolors="none", label=f"{g} (n={len(sub)})")
    ax.set_xscale("log")
    ax.set_xlabel("duration (ms, log scale)")
    ax.set_ylabel("mean frequency (kHz)")
    ax.legend(loc="upper right", fontsize=9)
    # marginal hists
    dmin, dmax = df["dur_ms"].min(), df["dur_ms"].max()
    bins_d = np.logspace(np.log10(max(dmin, 1e-2)), np.log10(dmax), 30)
    bins_f = np.linspace(df["avg_khz"].min(), df["avg_khz"].max(), 30)
    for g in ["WT", "Het"]:
        sub = df[df.geno == g]
        axtop.hist(sub["dur_ms"], bins=bins_d, color=C.GENO_COLOR[g], alpha=0.5,
                   density=True)
        axright.hist(sub["avg_khz"], bins=bins_f, color=C.GENO_COLOR[g], alpha=0.5,
                     density=True, orientation="horizontal")
    axtop.tick_params(labelbottom=False); axtop.set_yticks([])
    axright.tick_params(labelleft=False); axright.set_xticks([])
    for a in (axtop, axright):
        a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.suptitle("Duration vs mean frequency  (per call, session 6)", fontweight="bold")
    C.save_fig(fig, "fig03_dur_freq_scatter.png")
    plt.close(fig)


def main():
    df = load()
    res = print_stats(df)
    # honest flag on Het "calls"
    het = df[df.geno == "Het"]
    frac_min = (het["dur_ms"] <= het["dur_ms"].min() + 1e-6).mean()
    print("\n" + "-" * 78)
    print(f"CAVEAT: {frac_min*100:.0f}% of Het calls sit at the minimum detectable duration "
          f"({het['dur_ms'].min():.3f} ms).")
    print("        Such single-bin, short, low-contrast events may be broadband noise that")
    print("        passed the classifier rather than true tonal USVs. Interpret Het with care.")
    print("-" * 78)
    fig_violin(df, res)
    fig_by_phase(df)
    fig_scatter(df)


if __name__ == "__main__":
    main()
