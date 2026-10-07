"""Restyle the supplied legacy six-panel figure without replacing current analyses.

This reproduces the supplied 0-900 s image with 31097=WT / 31101=HET, confirmed
by the image and completed trial plan. The adjacent manifest has older inverted
labels. Separate cache and provenance distinguish the 15-minute image from the
matched-window analysis. Run this file in an IDE or use the
adjacent Windows launcher. --figures-only renders the validated numeric cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import lgdel_usv_analysis as analysis

# User configuration: numerical settings are separate from rendering settings.
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
MANIFEST = ROOT / "lgdel_usv_analysis" / "manifest.csv"
OUTPUT = ROOT / "lgdel_usv_analysis" / "style_revision"
ORIGINAL_PNG = ROOT / "lgdel_usv_analysis" / "lgdel_usv_wt_vs_het_summary.png"
END_SECONDS = 900.0
INTRO_SECONDS = 300.0
BIN_SECONDS = 30.0
CACHE_VERSION = 1
REFERENCE_LABELS = {"31097": "WT", "31101": "HET"}
EXPECTED_COUNTS = {("WT", "alone"): 68, ("HET", "alone"): 56,
                   ("WT", "partner"): 1786, ("HET", "partner"): 150}

COLORS = {"WT": "#32799D", "HET": "#DF892E"}
CLASS_COLORS = dict(zip(analysis.SYLLABLE_CLASSES, [
    "#80B6CF", "#9387B8", "#C886A3", "#6BAF98", "#D5B56D",
    "#65ADB3", "#C7CCD0", "#858FB6", "#92BCAD", "#BD975B", "#70A37C",
]))
FIGSIZE = (11.5, 12.8)
DPI = 350
BAR_HEIGHT = 0.24
BOX_WIDTH = 0.25
POINT_SIZE = 15
FONT_SIZE = 9
FORMATS = ("png", "pdf", "svg")
STEM = "lgdel_usv_wt_vs_het_clean"


def file_hash(path: Path) -> str:
    """Hash source contents so cache reuse cannot hide changed inputs."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_entries() -> list[analysis.CohortEntry]:
    """Read existing identities and resolve detector paths relative to this project.

    The two image-confirmed labels affect only in-memory copies for reproduction.
    Neither the manifest nor the current corrected analysis is overwritten.
    """
    rows = pd.read_csv(MANIFEST, dtype={"animal_id": str}).fillna("")
    entries = []
    for row in rows.itertuples():
        animal = row.animal_id
        session = DATA_ROOT / animal / "baseline" / "1"
        path = session / f"{animal}_1_baseline_outputs" / f"{animal}_1_baseline_stats.csv"
        if not path.exists():
            raise FileNotFoundError(f"Full-window detector table required: {path}")
        entries.append(analysis.CohortEntry(
            animal, REFERENCE_LABELS.get(animal, row.genotype), row.virus,
            session / f"{animal}_1_baseline.wav", path,
        ))
    return entries


def load_data(output: Path, recompute: bool, figures_only: bool):
    """Validate numeric provenance, reuse cache, or reproduce original aggregations."""
    entries = source_entries()
    fingerprint = {
        "version": CACHE_VERSION, "end_s": END_SECONDS, "intro_s": INTRO_SECONDS,
        "bin_s": BIN_SECONDS, "reference_labels": REFERENCE_LABELS,
        "sources": {str(e.stats_path): file_hash(e.stats_path) for e in entries},
        "manifest_sha256": file_hash(MANIFEST),
        "analysis_sha256": file_hash(Path(analysis.__file__)),
    }
    provenance_path = output / "provenance.json"
    names = ("calls", "summary", "bins", "statistics")
    valid = (provenance_path.exists()
             and json.loads(provenance_path.read_text())['fingerprint'] == fingerprint
             and all((output / f"{name}.csv").exists() for name in names))
    if figures_only and not valid:
        raise RuntimeError("No valid numeric cache. Run once without --figures-only.")
    if valid and not recompute:
        print("Validated cache reused; no numerical recomputation.")
        return tuple(pd.read_csv(output / f"{n}.csv", dtype={"animal_id": str}) for n in names)

    analysis.ANALYSIS_END_S = END_SECONDS
    analysis.PHASE_LIMITS = {"alone": (0, INTRO_SECONDS), "partner": (INTRO_SECONDS, END_SECONDS)}
    calls = analysis.load_calls(entries)
    counts = calls.groupby(["genotype", "phase"], observed=True).size().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Inputs do not reproduce supplied image: {counts}")
    summary = analysis.build_phase_summary(calls, entries)
    bins = analysis.build_time_bins(calls, entries, BIN_SECONDS)
    statistics = pd.concat([analysis.mann_whitney_table(summary),
                            analysis.repertoire_stats(calls)], ignore_index=True)
    data = (calls, summary, bins, statistics)
    for name, table in zip(names, data):
        table.to_csv(output / f"{name}.csv", index=False)
    provenance_path.write_text(json.dumps({
        "fingerprint": fingerprint,
        "purpose": "Styling reproduction of supplied 15-minute image, separate from matched-window analysis",
        "reference_png": str(ORIGINAL_PNG),
        "reference_sha256": file_hash(ORIGINAL_PNG),
        "animal_n": summary.groupby("genotype").animal_id.nunique().to_dict(),
        "statistics_note": "Original two-sided Mann-Whitney animal comparisons retained, unadjusted. "
            "Pooled-call chi-square composition tests are legacy annotations, not animal-level inference. "
            "Calls are nested within animals and several expected class counts are small. "
            "No new biological inference or corrected metadata is asserted by this restyling.",
    }, indent=2), encoding="utf-8")
    print(f"Reproduced supplied image: {counts}")
    return data


def style() -> None:
    """Use white backgrounds, fine dark spines, editable vector text and muted colors."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": FONT_SIZE, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": FONT_SIZE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#414141", "text.color": "#272727",
        "xtick.color": "#414141", "ytick.color": "#414141",
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.fontsize": 8, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def p_label(statistics: pd.DataFrame, phase: str, metric: str) -> str:
    """Keep the supplied image's statistical convention and original numeric p values."""
    p = analysis.lookup_p(statistics, phase, metric)
    return f"{analysis.p_text(p)} {analysis.p_stars(p)}"


def bracket(ax, x1: float, x2: float, fraction: float, text: str) -> None:
    """Reserve significance labels at a fixed fraction of the y axis range."""
    trans = ax.get_xaxis_transform()
    ax.plot([x1, x1, x2, x2], [fraction-.02, fraction, fraction, fraction-.02],
            transform=trans, color="#414141", lw=0.8, clip_on=False)
    ax.text((x1+x2)/2, fraction+.018, text, transform=trans,
            ha="center", va="bottom", fontsize=8)


def repertoire(ax, calls: pd.DataFrame, statistics: pd.DataFrame) -> None:
    """Draw four thin horizontal composition bars and two compact phase comparisons."""
    combos = [("WT", "alone"), ("HET", "alone"), ("WT", "partner"), ("HET", "partner")]
    y = np.array([3, 2.35, 1.15, .5])
    left = np.zeros(4)
    for cls in analysis.SYLLABLE_CLASSES:
        values = []
        for genotype, phase in combos:
            subset = calls[(calls.genotype == genotype) & (calls.phase == phase)]
            values.append((subset.class_top1 == cls).mean())
        ax.barh(y, values, left=left, height=BAR_HEIGHT, color=CLASS_COLORS[cls],
                edgecolor="white", linewidth=.25, label=cls.replace("_", " "))
        left += values
    ax.set(yticks=y, yticklabels=["WT, alone", "Het, alone", "WT, + partner", "Het, + partner"],
           xlim=(0, 1.02), ylim=(-.05, 3.55), xlabel="Syllable proportion",
           title="Syllable repertoire composition")
    ax.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"])
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    for yy, (genotype, phase) in zip(y, combos):
        n = len(calls[(calls.genotype == genotype) & (calls.phase == phase)])
        ax.text(1.035, yy, f"n={n}", va="center", fontsize=7.5)
    for yy, phase in [(3.36, "alone"), (1.51, "partner")]:
        ax.text(.5, yy, p_label(statistics, phase, "class_top1"), ha="center", fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(-.02, -.23), ncol=4,
              fontsize=6.8, handlelength=1.2, columnspacing=.85, labelspacing=.6)


def boxstrip(ax, summary: pd.DataFrame, statistics: pd.DataFrame) -> None:
    """Show all 12 animals, IQR boxes, medians and white mean diamonds on a linear scale."""
    for i, phase in enumerate(analysis.PHASE_ORDER):
        for genotype, offset in [("WT", -.18), ("HET", .18)]:
            values = summary.loc[(summary.genotype == genotype) & (summary.phase == phase),
                                 "call_rate_per_min"].to_numpy(float)
            center = i + offset
            ax.boxplot(values, positions=[center], widths=BOX_WIDTH, patch_artist=True,
                       showfliers=False, manage_ticks=False,
                       boxprops={"facecolor": COLORS[genotype], "edgecolor": "#414141", "linewidth": .8},
                       medianprops={"color": "white", "linewidth": 1.2},
                       whiskerprops={"color": "#414141", "linewidth": .8},
                       capprops={"color": "#414141", "linewidth": .8})
            # Deterministic spreading makes exact ties visible without changing values.
            jitter = np.zeros(len(values))
            for val in np.unique(values):
                indices = np.flatnonzero(values == val)
                jitter[indices] = np.linspace(-.085, .085, len(indices)) if len(indices)>1 else 0
            ax.scatter(center+jitter, values, s=POINT_SIZE, facecolor="white",
                       edgecolor="#414141", linewidth=.6, zorder=4)
            ax.scatter(center, values.mean(), marker="D", s=33,
                       facecolor="white", edgecolor="#414141", linewidth=.8, zorder=5)
        bracket(ax, i-.3, i+.3, .85 if i == 0 else .95,
                p_label(statistics, phase, "call_rate_per_min"))
    ax.set(xlim=(-.55, 1.55), ylim=(-1, summary.call_rate_per_min.max()*1.22),
           xticks=[0, 1], xticklabels=["alone", "+ partner"], ylabel="Calls / min",
           title="Call rate by genotype and phase")
    handles = [Line2D([], [], color=COLORS[g], lw=5, label=f"{analysis.GENOTYPE_LABEL[g]} (n=12)")
               for g in analysis.GENOTYPE_ORDER]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(.01, .62), fontsize=7)


def render(data, output: Path) -> None:
    """Render the original six panels with the requested visual treatment."""
    calls, summary, bins, statistics = data
    style()
    analysis.GENOTYPE_COLOR = COLORS
    analysis.ANALYSIS_END_S = END_SECONDS
    fig = plt.figure(figsize=FIGSIZE)
    grid = fig.add_gridspec(4, 2, height_ratios=[1.15, 1, .95, 1],
                           left=.105, right=.965, top=.935, bottom=.065,
                           wspace=.40, hspace=.72)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]),
            fig.add_subplot(grid[2, :]), fig.add_subplot(grid[3, :])]
    repertoire(axes[0], calls, statistics)
    analysis.plot_total_calls(axes[1], summary)
    for patch in axes[1].patches:
        patch.set_width(.64)
    boxstrip(axes[2], summary, statistics)
    analysis.plot_acoustics(axes[3], calls)
    analysis.plot_timeline(axes[4], bins)
    analysis.plot_vocal_output(axes[5], summary, statistics)
    for i, ax in enumerate(axes):
        ax.grid(False)
        ax.set_title(ax.get_title(), pad=13)
        ax.text(-.16 if i < 4 else -.065, 1.10, chr(97+i),
                transform=ax.transAxes, fontweight="bold", fontsize=12)
    # Keep the last panel's legacy p values but reserve space above individual traces.
    for text in list(axes[5].texts):
        if text.get_text().startswith(("ns", "*")):
            text.remove()
    for line in list(axes[5].lines):
        if line.get_color() == "#333333":
            line.remove()
    axes[5].set_ylim(-.03, summary.vocal_output_s_per_min.max()*1.18)
    axes[5].set_xlim(-.16, 1.16)
    for i, phase in enumerate(analysis.PHASE_ORDER):
        bracket(axes[5], i-.09, i+.09, .88,
                p_label(statistics, phase, "vocal_output_s_per_min"))
    axes[5].legend(loc="center left", bbox_to_anchor=(.005, .65), ncol=2)
    axes[4].set_ylim(bottom=0)
    fig.suptitle("LgDel USV analysis: WT vs Het", y=.985, fontsize=15, fontweight="bold")
    fig.text(.5, .960, "Grouped across virus  |  partner introduced at 5 min",
             ha="center", fontsize=9, color="#666666")
    for suffix in FORMATS:
        fig.savefig(output / f"{STEM}.{suffix}", dpi=DPI, facecolor="white")
    plt.close(fig)


def comparison(output: Path) -> None:
    """Export a visual benchmark against the untouched original figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
    for ax, path, title in zip(axes, [ORIGINAL_PNG, output / f"{STEM}.png"],
                                ["Original", "Revised style"]):
        ax.imshow(plt.imread(path))
        ax.set_title(title)
        ax.axis("off")
    for suffix in FORMATS:
        fig.savefig(output / f"style_comparison.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    """Provide explicit cache controls and keep all outputs in a separate directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--recompute", action="store_true")
    mode.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.output_dir, args.recompute, args.figures_only)
    render(data, args.output_dir)
    comparison(args.output_dir)
    print(f"Exports: {args.output_dir / STEM}")


if __name__ == "__main__":
    main()
