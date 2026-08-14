"""MODULE B - Syllable repertoire (class_top1 among 11 types) WT vs Het, per phase.

Outputs:
  fig02_repertoire_stacked.png : stacked class-proportion bars per (geno x phase).
  fig02_class_heatmap.png      : rows=11 classes, cols=(geno,phase), within-column proportion.
  fig02_class_counts.png       : class counts by genotype (grouped bars, WT blue / Het orange).

Stats:
  chi-square of class distribution WT vs Het (pooled social phases: female + male).
  Shannon entropy of the repertoire per genotype.

HONESTY: WT emit ~7x more calls than Het here (559 vs 77). The 'alone' phase and all Het
phases have single/low-double-digit n. The Het repertoire is ~90% 'short', which is exactly
the signature of broadband noise passing the classifier, so Het class labels are unreliable.
"""
import sys
sys.path.insert(0, "/home/andry/UVS/scripts")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import common as C

C.apply_style()

CLASSES = C.SYLLABLE_CLASSES  # 11-class canonical ordering
# distinct qualitative colours for the 11 syllable classes
CLASS_COLORS = dict(zip(CLASSES, plt.cm.tab20(np.linspace(0, 1, len(CLASSES)))))


def shannon(counts):
    p = np.asarray(counts, float)
    p = p[p > 0]
    if p.sum() == 0:
        return np.nan
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    df = C.load_calls()
    # attach class label from the meta csv (same row order guaranteed by design)
    meta = pd.read_csv(C.SCRIPTS / "calls_meta.csv")
    df["class_top1"] = meta["class_top1"].values
    df["class_top1"] = pd.Categorical(df["class_top1"], categories=CLASSES, ordered=True)

    # ---- count table: rows=class, cols=(geno,phase) ------------------------
    cols = [(g, p) for g in ("WT", "Het") for p in C.PHASE_ORDER]
    counts = pd.DataFrame(0, index=CLASSES, columns=pd.MultiIndex.from_tuples(cols))
    for (g, p), sub in df.groupby(["geno", "phase"], observed=True):
        vc = sub["class_top1"].value_counts()
        for cls in CLASSES:
            counts.loc[cls, (g, p)] = int(vc.get(cls, 0))
    col_tot = counts.sum(axis=0).replace(0, np.nan)
    prop = counts.div(col_tot, axis=1).fillna(0.0)   # within-column proportion

    print("=== counts (class x geno,phase) ===")
    print(counts)
    print("\ncolumn totals (n calls):")
    print(col_tot)

    # =====================================================================
    # FIG 1 : stacked bars of class proportion per (geno x phase) column
    # =====================================================================
    fig, ax = plt.subplots(figsize=(9, 6))
    xlab = [f"{g}\n{C.PHASE_LABEL[p]}" for (g, p) in cols]
    x = np.arange(len(cols))
    bottom = np.zeros(len(cols))
    for cls in CLASSES:
        vals = prop[cols].loc[cls].values.astype(float)
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottom, width=0.72, color=CLASS_COLORS[cls],
               edgecolor="white", linewidth=0.4, label=cls)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(xlab)
    ax.set_ylabel("within-column proportion of calls")
    ax.set_ylim(0, 1)
    ax.set_title("Syllable repertoire composition (WT vs Het x phase)", pad=24)
    # annotate n on top of each bar
    for xi, (g, p) in zip(x, cols):
        n = int(counts[(g, p)].sum())
        ax.text(xi, 1.02, f"n={n}", ha="center", va="bottom", fontsize=9, color="#333333")
    # separator between WT and Het blocks
    ax.axvline(2.5, color="#999999", lw=1.0, ls="--")
    ax.grid(axis="x", visible=False)
    ax.legend(title="class", bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize=9)
    C.save_fig(fig, "fig02_repertoire_stacked.png")
    plt.close(fig)

    # =====================================================================
    # FIG 2 : heatmap rows=11 classes, cols=(geno,phase), within-col proportion
    # =====================================================================
    fig, ax = plt.subplots(figsize=(8, 7))
    M = prop[cols].values.astype(float)
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([f"{g}\n{C.PHASE_LABEL[p]}" for (g, p) in cols])
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.set_title("Class proportion heatmap (within-column)")
    ax.axvline(2.5, color="#66ccff", lw=1.5)
    for i in range(len(CLASSES)):
        for j in range(len(cols)):
            v = M[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v < 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("proportion within column")
    ax.grid(False)
    C.save_fig(fig, "fig02_class_heatmap.png")
    plt.close(fig)

    # =====================================================================
    # FIG 3 : class counts by genotype (grouped bars, WT blue / Het orange)
    # =====================================================================
    wt_counts = counts[[("WT", p) for p in C.PHASE_ORDER]].sum(axis=1)
    het_counts = counts[[("Het", p) for p in C.PHASE_ORDER]].sum(axis=1)
    # order classes by total prevalence for readability
    order = (wt_counts + het_counts).sort_values(ascending=False).index.tolist()
    wt_o = wt_counts[order].values
    het_o = het_counts[order].values

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(order))
    w = 0.4
    ax.bar(x - w / 2, wt_o, width=w, color=C.GENO_COLOR["WT"], label=f"WT (n={int(wt_o.sum())})")
    ax.bar(x + w / 2, het_o, width=w, color=C.GENO_COLOR["Het"], label=f"Het (n={int(het_o.sum())})")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=40, ha="right")
    ax.set_ylabel("call count")
    ax.set_title("Syllable-class counts by genotype (all phases pooled)")
    ax.grid(axis="x", visible=False)
    ax.legend()
    C.save_fig(fig, "fig02_class_counts.png")
    plt.close(fig)

    # =====================================================================
    # STATS
    # =====================================================================
    print("\n=== STATS ===")
    # chi-square: class distribution WT vs Het, pooled social phases (female+male)
    soc = df[df["phase"].isin(["female", "male"])]
    ct = pd.crosstab(soc["class_top1"], soc["geno"])
    ct = ct.reindex(index=CLASSES).fillna(0)
    ct = ct.loc[(ct.sum(axis=1) > 0)]   # drop all-zero classes for a valid test
    chi2, pchi, dof, _ = chi2_contingency(ct.values)
    n_small = int((chi2_contingency(ct.values)[3] < 5).sum())
    print("contingency (social phases pooled), class x geno:")
    print(ct)
    print(f"chi2={chi2:.3f}  dof={dof}  p={pchi:.3e}  "
          f"(expected cells <5: {n_small}/{ct.size} -> chi2 approximation is shaky)")

    # Shannon entropy of repertoire per genotype (all phases pooled)
    ent = {}
    for g in ("WT", "Het"):
        c = counts[[(g, p) for p in C.PHASE_ORDER]].sum(axis=1).values
        ent[g] = shannon(c)
    print(f"Shannon entropy (bits): WT={ent['WT']:.3f}  Het={ent['Het']:.3f}  "
          f"(max possible = log2(11) = {np.log2(11):.3f})")

    # dominant classes per genotype
    dom = {}
    for g, cser in (("WT", wt_counts), ("Het", het_counts)):
        tot = cser.sum()
        top = cser.sort_values(ascending=False).head(3)
        dom[g] = [(cls, int(n), n / tot) for cls, n in top.items()]
        print(f"dominant {g}: " + ", ".join(f"{c}={n}({f:.0%})" for c, n, f in dom[g]))

    return dict(counts=counts, chi2=chi2, pchi=pchi, dof=dof, ent=ent, dom=dom,
                col_tot=col_tot)


if __name__ == "__main__":
    main()
