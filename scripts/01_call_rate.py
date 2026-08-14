"""MODULE A - Vocal output across phases (WT vs Het).

Session 6 = 900 s, three 300-s phases:
  alone  [  0-300):  resident male alone
  female [300-600):  a female is introduced
  male   [600-900):  a second (intruder) male is added

Produces three figures and prints animal-level statistics.
Because each phase lasts exactly 300 s = 5 min, calls/min = (# calls) / 5.

Honesty notes printed at the end:
  * WT emit ~10-15x more calls than Het.
  * n is tiny: 2 WT mice vs 4 Het mice. The 'alone' phase in particular has
    single-digit call counts for several animals.
  * Some Het 'calls' may be broadband noise that slipped past the classifier.
"""
import sys
sys.path.insert(0, "/home/andry/UVS/scripts")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common as C

C.apply_style()

PHASE_S = 300.0          # seconds per phase
PHASE_MIN = PHASE_S / 60  # = 5 minutes
BIN_S = 30.0             # timeline bin width
SESSION_S = 900.0

df = C.load_calls()
mice = [e["mouse"] for e in C.FILES]
geno_of = {e["mouse"]: e["geno"] for e in C.FILES}
wt_mice = [m for m in mice if geno_of[m] == "WT"]
het_mice = [m for m in mice if geno_of[m] == "Het"]

# ---------------------------------------------------------------------------
# Per-mouse x per-phase call counts -> calls/min (300 s -> /5)
# ---------------------------------------------------------------------------
counts = (df.groupby(["mouse", "phase"], observed=False).size()
            .unstack(fill_value=0)
            .reindex(index=mice, columns=C.PHASE_ORDER, fill_value=0))
rate = counts / PHASE_MIN     # calls per minute, per mouse per phase
totals = counts.sum(axis=1)   # total calls per mouse

print("\n=== Per-mouse call counts by phase (session 6) ===")
for m in mice:
    print(f"  {m} ({geno_of[m]:>3}):  alone={counts.loc[m,'alone']:3d}  "
          f"female={counts.loc[m,'female']:3d}  male={counts.loc[m,'male']:3d}  "
          f"total={totals.loc[m]:3d}")

# group-level calls/min per phase (mean over animals)
grp_rate = {}
for g, gm in (("WT", wt_mice), ("Het", het_mice)):
    grp_rate[g] = {ph: rate.loc[gm, ph].to_numpy() for ph in C.PHASE_ORDER}

print(f"\nn: WT={len(wt_mice)} mice, Het={len(het_mice)} mice")
print("\n=== Group mean calls/min (mean +/- sem over animals) ===")
mean_rate = {g: {} for g in ("WT", "Het")}
for g in ("WT", "Het"):
    for ph in C.PHASE_ORDER:
        v = grp_rate[g][ph]
        mean_rate[g][ph] = float(np.mean(v))
        sem = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        print(f"  {g:>3} {ph:>6}: mean={np.mean(v):6.2f}  sem={sem:6.2f}  "
              f"(per-mouse: {np.array2string(v, precision=2)})")

# ---------------------------------------------------------------------------
# Stats: WT vs Het calls/min per phase; within-genotype alone->female change
# ---------------------------------------------------------------------------
print("\n=== WT vs Het calls/min per phase (Mann-Whitney, animal-level) ===")
pvals_wt_het = {}
for ph in C.PHASE_ORDER:
    U, p, rbc = C.mannwhitney(grp_rate["WT"][ph], grp_rate["Het"][ph])
    pvals_wt_het[ph] = p
    print(f"  {ph:>6}: U={U}  p={p:.4g}  rank-biserial={rbc:+.3f}  {C.pstars(p)}")

print("\n=== Within-genotype alone -> female (paired per mouse) ===")
fold = {}
for g, gm in (("WT", wt_mice), ("Het", het_mice)):
    a = rate.loc[gm, "alone"].to_numpy()
    f = rate.loc[gm, "female"].to_numpy()
    # fold-increase on group means (guard divide-by-zero)
    ma, mf = float(np.mean(a)), float(np.mean(f))
    fold[g] = (mf / ma) if ma > 0 else np.inf
    # Wilcoxon signed-rank where possible (n small; also report raw)
    try:
        from scipy.stats import wilcoxon
        if np.any(a != f):
            W, pw = wilcoxon(a, f)
        else:
            W, pw = np.nan, np.nan
    except Exception:
        W, pw = np.nan, np.nan
    print(f"  {g:>3}: alone mean={ma:.2f}/min  female mean={mf:.2f}/min  "
          f"fold(female/alone)={fold[g]:.2f}x  Wilcoxon W={W} p={pw if isinstance(pw,float) else pw}")

# ===========================================================================
# FIG 01a - calls/min by phase, per-mouse points + group mean
# ===========================================================================
fig, ax = plt.subplots(figsize=(6.4, 4.6))
x = np.arange(len(C.PHASE_ORDER))
rng = np.random.default_rng(0)
for g, gm in (("WT", wt_mice), ("Het", het_mice)):
    col = C.GENO_COLOR[g]
    # per-mouse points (jittered)
    for m in gm:
        y = [rate.loc[m, ph] for ph in C.PHASE_ORDER]
        jx = x + (rng.uniform(-0.10, 0.10) + (0.14 if g == "Het" else -0.14))
        ax.plot(jx, y, "-", color=col, alpha=0.30, lw=1.0, zorder=1)
        ax.scatter(jx, y, s=34, color=col, alpha=0.65, edgecolor="white",
                   linewidth=0.6, zorder=2)
    # group mean +/- sem
    means = np.array([mean_rate[g][ph] for ph in C.PHASE_ORDER])
    sems = np.array([np.std(grp_rate[g][ph], ddof=1) / np.sqrt(len(gm))
                     if len(gm) > 1 else 0.0 for ph in C.PHASE_ORDER])
    off = 0.14 if g == "Het" else -0.14
    ax.errorbar(x + off, means, yerr=sems, color=col, lw=2.4, marker="o",
                ms=8, capsize=4, zorder=3,
                label=f"{g} (n={len(gm)})", markeredgecolor="white")
# significance brackets (WT vs Het) per phase
ymax = rate.to_numpy().max()
for i, ph in enumerate(C.PHASE_ORDER):
    C.sig_bracket(ax, i - 0.14, i + 0.14, ymax * (1.04 + 0.06 * i), pvals_wt_het[ph])
ax.set_xticks(x)
ax.set_xticklabels([C.PHASE_LABEL[p] for p in C.PHASE_ORDER])
ax.set_ylabel("call rate (calls / min)")
ax.set_title("Vocal output by social phase (session 6)")
ax.legend(loc="center right")
ax.set_xlim(-0.5, len(C.PHASE_ORDER) - 0.5)
fig.tight_layout()
C.save_fig(fig, "fig01_callrate_by_phase.png")
plt.close(fig)

# ===========================================================================
# FIG 01b - timeline: calls per 30 s bin, mean +/- sem per genotype
# ===========================================================================
edges = np.arange(0, SESSION_S + BIN_S, BIN_S)
centers = 0.5 * (edges[:-1] + edges[1:])
# per-mouse histogram of call start times
hist = {}
for m in mice:
    t = df.loc[df["mouse"] == m, "start(s)"].to_numpy()
    hist[m], _ = np.histogram(t, bins=edges)

fig, ax = plt.subplots(figsize=(8.4, 4.4))
for g, gm in (("WT", wt_mice), ("Het", het_mice)):
    col = C.GENO_COLOR[g]
    H = np.vstack([hist[m] for m in gm]).astype(float)  # [mice, bins]
    mean = H.mean(axis=0)
    sem = H.std(axis=0, ddof=1) / np.sqrt(H.shape[0]) if H.shape[0] > 1 else np.zeros_like(mean)
    ax.plot(centers, mean, color=col, lw=2.0, label=f"{g} (n={len(gm)})")
    ax.fill_between(centers, mean - sem, mean + sem, color=col, alpha=0.20)
for b, lab in zip(C.PHASE_BOUNDS[:-1], ["+ female", "+ 2nd male"]):
    ax.axvline(b, color="#555555", ls="--", lw=1.1)
    ax.text(b + 6, ax.get_ylim()[1] * 0.94, lab, fontsize=9, color="#555555")
ax.text(6, ax.get_ylim()[1] * 0.94, "alone", fontsize=9, color="#555555")
ax.set_xlabel("time in session (s)")
ax.set_ylabel(f"calls / {int(BIN_S)} s bin (mean +/- sem)")
ax.set_title("Call timeline: female introduction surge")
ax.set_xlim(0, SESSION_S)
ax.legend(loc="upper right")
fig.tight_layout()
C.save_fig(fig, "fig01_timeline.png")
plt.close(fig)

# ===========================================================================
# FIG 01c - total output per mouse (bars) + calls/min per genotype per phase
# ===========================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                               gridspec_kw={"width_ratios": [1.1, 1.0]})
# left: total calls per mouse
order = wt_mice + het_mice
bx = np.arange(len(order))
bcol = [C.GENO_COLOR[geno_of[m]] for m in order]
axL.bar(bx, [totals.loc[m] for m in order], color=bcol, edgecolor="white", linewidth=0.8)
for i, m in enumerate(order):
    axL.text(i, totals.loc[m] + max(totals) * 0.01, str(int(totals.loc[m])),
             ha="center", va="bottom", fontsize=9)
axL.set_xticks(bx)
axL.set_xticklabels(order, rotation=30, ha="right")
axL.set_ylabel("total calls (session 6)")
axL.set_title("Total vocal output per mouse")
# proxy legend
from matplotlib.patches import Patch
axL.legend(handles=[Patch(color=C.GENO_COLOR["WT"], label="WT"),
                    Patch(color=C.GENO_COLOR["Het"], label="Het")], loc="upper right")

# right: grouped bars calls/min per genotype per phase (mean +/- sem)
gx = np.arange(len(C.PHASE_ORDER))
w = 0.36
for k, (g, gm) in enumerate((("WT", wt_mice), ("Het", het_mice))):
    means = np.array([mean_rate[g][ph] for ph in C.PHASE_ORDER])
    sems = np.array([np.std(grp_rate[g][ph], ddof=1) / np.sqrt(len(gm))
                     if len(gm) > 1 else 0.0 for ph in C.PHASE_ORDER])
    axR.bar(gx + (k - 0.5) * w, means, w, yerr=sems, capsize=3,
            color=C.GENO_COLOR[g], edgecolor="white", linewidth=0.8,
            label=f"{g} (n={len(gm)})")
axR.set_xticks(gx)
axR.set_xticklabels([C.PHASE_LABEL[p] for p in C.PHASE_ORDER])
axR.set_ylabel("call rate (calls / min)")
axR.set_title("Call rate per genotype x phase")
axR.legend(loc="upper left")
fig.tight_layout()
C.save_fig(fig, "fig01_total_output.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Machine-readable summary
# ---------------------------------------------------------------------------
print("\n=== SUMMARY (calls/min group means) ===")
for g in ("WT", "Het"):
    print(f"  {g}: alone={mean_rate[g]['alone']:.2f}  female={mean_rate[g]['female']:.2f}  "
          f"male={mean_rate[g]['male']:.2f}  |  fold alone->female={fold[g]:.2f}x")
print("  p (WT vs Het): " + "  ".join(f"{ph}={pvals_wt_het[ph]:.4g}" for ph in C.PHASE_ORDER))
print(f"\n  Totals: WT={int(totals[wt_mice].sum())} calls, Het={int(totals[het_mice].sum())} calls; "
      f"ratio WT/Het = {totals[wt_mice].sum()/max(1,totals[het_mice].sum()):.1f}x")
print("\nHONESTY: n=2 WT vs 4 Het; Mann-Whitney cannot reach p<0.05 with n=2 vs 4 "
      "(min two-sided p ~ 0.13). Treat all p-values as underpowered. Some Het calls may be noise.")
