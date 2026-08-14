"""MODULE E - CAN WE SEPARATE THE VOICES? (Q4, exploratory).

The hard "who-called" problem from a SINGLE microphone with NO ground truth.
We focus on the two WT resident males (31096, 31097) that emit enough calls.

Per WT mouse, using its own DINOv2 spectrogram embeddings:

1. NOVELTY / ANOMALY proxy for newcomer vocalisations.
   - Build a RESIDENT reference = alone-phase calls + the earliest few female-phase
     calls (the resident is almost certainly the caller right when the female enters).
   - Fit a kNN cosine-distance novelty model on that reference.
   - Score female- and male-phase calls; the fraction scoring "novel" (distance above
     the reference's own 95th LOO percentile) is a WEAK proxy for newcomer calls.

2. UNSUPERVISED CLUSTERING in embedding space (KMeans k=2..4, silhouette-picked).
   - Cross-tabulate cluster x phase to see whether NEW acoustic types appear once the
     female / 2nd male are added (would hint at newcomer voices), or whether clusters
     are phase-agnostic (all one resident voice, just more calls).

STRONG CAVEATS (printed & drawn): single microphone, no ground-truth speaker labels;
female mice emit very few USVs so MOST calls are the resident male regardless of phase;
the alone-phase anchor is tiny (n=6 and n=3); this is HYPOTHESIS-GENERATING, not
validated voice attribution.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/andry/UVS/scripts")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency
import umap

import common as C
C.apply_style()

RNG = 0
WT_MICE = ["31096", "31097"]
PHASE_COL = {"alone": "#6c6c6c", "female": "#d1495b", "male": "#2a9d8f"}
N_FEMALE_ANCHOR = 12   # earliest female calls treated as still-resident reference
KNN_K = 5              # neighbours for the kNN novelty distance


# ---------------------------------------------------------------------------
def load_wt():
    """Return dict mouse -> DataFrame(row index into embedding array) for WT mice."""
    d = np.load(C.SCRIPTS / "calls_dataset.npz", allow_pickle=True)
    emb = d["emb_dinov2"].astype(np.float64)
    df = pd.DataFrame({
        "mouse": d["mouse"].astype(str),
        "phase": d["phase"].astype(str),
        "top1": d["top1"].astype(str),
        "start": d["start"].astype(float),
    })
    df["row"] = np.arange(len(df))
    return emb, df


def knn_novelty(emb, sub, mouse):
    """kNN cosine-distance novelty scores for one mouse's calls.

    Reference = alone calls + earliest N_FEMALE_ANCHOR female calls.
    Returns per-call novelty score (mean cosine dist to k nearest reference points),
    the reference-derived 95th-percentile threshold, and the reference mask.
    """
    X = normalize(emb[sub["row"].to_numpy()])          # L2 -> cosine via Euclidean
    phase = sub["phase"].to_numpy()
    start = sub["start"].to_numpy()

    ref_mask = phase == "alone"
    fem_idx = np.where(phase == "female")[0]
    fem_order = fem_idx[np.argsort(start[fem_idx])]
    ref_mask[fem_order[:N_FEMALE_ANCHOR]] = True

    Xref = X[ref_mask]
    n_ref = len(Xref)
    k = min(KNN_K, n_ref - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Xref)
    # leave-one-out distances inside the reference (drop the self=0 neighbour)
    dref, _ = nn.kneighbors(Xref)
    ref_score = dref[:, 1:].mean(axis=1)
    thr = np.percentile(ref_score, 95)

    # score every call (reference points included; nearest may be self -> drop col 0
    #  only for reference rows). Simpler: query k neighbours, use them directly.
    nn_score = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Xref)
    dall, _ = nn_score.kneighbors(X)
    score = dall.mean(axis=1)
    # for reference rows the self-point (dist 0) inflates "non-novelty"; recompute
    # them with the LOO score to keep it honest
    score[ref_mask] = ref_score
    return score, thr, ref_mask, n_ref, k


def pick_kmeans(Xn, krange=(2, 3, 4)):
    """KMeans over k, pick best silhouette (cosine on L2-normed embeddings)."""
    best = None
    for k in krange:
        km = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit(Xn)
        sil = silhouette_score(Xn, km.labels_, metric="cosine")
        if best is None or sil > best[0]:
            best = (sil, k, km.labels_)
    return best  # (silhouette, k, labels)


# ---------------------------------------------------------------------------
def main():
    emb, df = load_wt()
    results = {}

    # ------- per-mouse compute -------------------------------------------
    per = {}
    for m in WT_MICE:
        sub = df[df["mouse"] == m].reset_index(drop=True)
        X = emb[sub["row"].to_numpy()]
        Xn = normalize(X)

        score, thr, ref_mask, n_ref, k = knn_novelty(emb, sub, m)
        sub = sub.assign(novelty=score, is_ref=ref_mask)

        # novel fraction per phase
        novel = {}
        for ph in C.PHASE_ORDER:
            s = sub.loc[sub["phase"] == ph, "novelty"].to_numpy()
            novel[ph] = (float(np.mean(s > thr)) if len(s) else np.nan, len(s))

        # UMAP embedding (2D)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                            random_state=RNG)
        xy = reducer.fit_transform(Xn)

        # clustering + cluster x phase crosstab
        sil, kbest, labels = pick_kmeans(Xn)
        sub = sub.assign(cluster=labels)
        ct = pd.crosstab(sub["cluster"], sub["phase"]).reindex(
            columns=C.PHASE_ORDER, fill_value=0)
        # chi-square: is cluster membership associated with phase?
        try:
            chi2, pchi, _, _ = chi2_contingency(ct.to_numpy())
        except ValueError:
            chi2, pchi = np.nan, np.nan
        # column-normalised (within each phase, fraction in each cluster)
        ct_colnorm = ct.div(ct.sum(axis=0).replace(0, np.nan), axis=1)

        per[m] = dict(sub=sub, xy=xy, thr=thr, novel=novel, n_ref=n_ref, k=k,
                      sil=sil, kbest=kbest, ct=ct, ct_colnorm=ct_colnorm,
                      chi2=chi2, pchi=pchi)
        results[m] = dict(novel=novel, kbest=kbest, sil=float(sil),
                          pchi=float(pchi) if np.isfinite(pchi) else None,
                          n_ref=n_ref)

    # ================= FIG 1: UMAP per mouse by phase =====================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, m in zip(axes, WT_MICE):
        p = per[m]; sub = p["sub"]; xy = p["xy"]
        for ph in C.PHASE_ORDER:
            mk = (sub["phase"] == ph).to_numpy()
            ax.scatter(xy[mk, 0], xy[mk, 1], s=22, c=PHASE_COL[ph],
                       alpha=0.6, edgecolors="none", label=C.PHASE_LABEL[ph])
        # alone anchors highlighted
        am = (sub["phase"] == "alone").to_numpy()
        ax.scatter(xy[am, 0], xy[am, 1], s=120, marker="*",
                   facecolors="none", edgecolors="black", linewidths=1.4,
                   label="alone anchor", zorder=5)
        ax.set_title(f"WT {m}  (n={len(sub)} calls)", color=C.WT_COLOR)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.legend(loc="best", fontsize=8, markerscale=0.9)
    fig.suptitle("DINOv2 embedding UMAP per WT resident, coloured by phase",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    C.save_fig(fig, "fig05_attribution_umap.png")
    plt.close(fig)

    # ================= FIG 2: novelty distributions =======================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, m in zip(axes, WT_MICE):
        p = per[m]; sub = p["sub"]; thr = p["thr"]
        data = [sub.loc[sub["phase"] == ph, "novelty"].to_numpy()
                for ph in C.PHASE_ORDER]
        parts = ax.violinplot(data, positions=range(3), showextrema=False,
                              widths=0.8)
        for pc, ph in zip(parts["bodies"], C.PHASE_ORDER):
            pc.set_facecolor(PHASE_COL[ph]); pc.set_alpha(0.35)
        for i, (d, ph) in enumerate(zip(data, C.PHASE_ORDER)):
            jit = np.random.RandomState(RNG).normal(0, 0.05, len(d))
            ax.scatter(np.full(len(d), i) + jit, d, s=10, c=PHASE_COL[ph],
                       alpha=0.6, edgecolors="none")
        ax.axhline(thr, ls="--", lw=1.2, c="black",
                   label=f"novel thr (ref p95={thr:.3f})")
        # annotate novel fraction for female / male
        ymax = ax.get_ylim()[1]
        for i, ph in enumerate(C.PHASE_ORDER):
            frac, n = p["novel"][ph]
            txt = f"{frac*100:.0f}%\nnovel\n(n={n})" if np.isfinite(frac) else f"n={n}"
            ax.text(i, ymax * 0.98, txt, ha="center", va="top", fontsize=8)
        ax.set_xticks(range(3))
        ax.set_xticklabels([C.PHASE_LABEL[ph] for ph in C.PHASE_ORDER])
        ax.set_ylabel("kNN cosine novelty score")
        ax.set_title(f"WT {m}  (ref n={p['n_ref']}, k={p['k']})",
                     color=C.WT_COLOR)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Novelty vs resident reference (alone + earliest female anchors)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    C.save_fig(fig, "fig05_novelty.png")
    plt.close(fig)

    # ================= FIG 3: cluster x phase heatmaps ====================
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, m in zip(axes, WT_MICE):
        p = per[m]; ctn = p["ct_colnorm"]
        im = ax.imshow(ctn.to_numpy(), aspect="auto", cmap="magma",
                       vmin=0, vmax=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels([C.PHASE_LABEL[ph] for ph in C.PHASE_ORDER])
        ax.set_yticks(range(ctn.shape[0]))
        ax.set_yticklabels([f"cl {c}" for c in ctn.index])
        ax.set_title(f"WT {m}  k={p['kbest']}, sil={p['sil']:.2f}\n"
                     f"chi2 p={p['pchi']:.1e}", color=C.WT_COLOR, fontsize=10)
        # annotate raw counts
        raw = p["ct"].to_numpy()
        for i in range(ctn.shape[0]):
            for j in range(3):
                v = ctn.to_numpy()[i, j]
                ax.text(j, i, f"{raw[i, j]}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="frac of phase in cluster")
    fig.suptitle("Cluster x phase cross-tab (counts; colour = column-normalised)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    C.save_fig(fig, "fig05_cluster_phase.png")
    plt.close(fig)

    # ------- console report ----------------------------------------------
    print("\n" + "=" * 70)
    print("MODULE E  -  VOICE ATTRIBUTION (exploratory, no ground truth)")
    print("=" * 70)
    for m in WT_MICE:
        p = per[m]
        print(f"\nWT {m}:  reference n={p['n_ref']} (alone + earliest {N_FEMALE_ANCHOR} female), "
              f"kNN k={p['k']}, novelty p95 thr={p['thr']:.3f}")
        for ph in C.PHASE_ORDER:
            frac, n = p["novel"][ph]
            fs = f"{frac*100:5.1f}%" if np.isfinite(frac) else "  n/a"
            print(f"    {ph:7s}: novel fraction {fs}   (n={n})")
        print(f"  KMeans best k={p['kbest']} (silhouette={p['sil']:.3f}); "
              f"cluster~phase chi2 p={p['pchi']:.2e}")
        print(f"  cluster x phase counts:\n{p['ct'].to_string()}")
        # verdict on whether clusters track phase
        track = (np.isfinite(p['pchi']) and p['pchi'] < 0.05)
        print(f"  clusters track phase? {'YES (assoc. sig.)' if track else 'NO (phase-agnostic)'}")
    print("\nCAVEATS: single mic, NO ground truth; female USVs are rare so most calls")
    print("are the resident male regardless of phase; alone anchor tiny; this is a")
    print("hypothesis-generating estimate, NOT validated speaker attribution.")
    return results


if __name__ == "__main__":
    main()
