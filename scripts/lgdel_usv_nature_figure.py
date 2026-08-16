"""Create a compact Nature-style figure for matched-window LgDel USV results.

The figure emphasizes animal-level inference:

* paired pre/post box-and-strip plots for call rate and vocal output;
* animal-level post-minus-pre change scores for the genotype interaction;
* a median and interquartile-range time course that is robust to high callers.

Run after ``lgdel_usv_analysis.py`` has generated the matched 5-minute tables.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


WT_COLOR = "#356FA3"
HET_COLOR = "#D6752B"
PALETTE = {"WT": WT_COLOR, "HET": HET_COLOR}
GENOTYPE_ORDER = ["WT", "HET"]
PHASE_ORDER = ["alone", "partner"]
PHASE_LABELS = ["Alone", "+ partner"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "lgdel_usv_analysis_matched_5min",
    )
    parser.add_argument(
        "--output-stem",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1]
        / "lgdel_usv_analysis_matched_5min"
        / "lgdel_usv_wt_vs_het_nature",
    )
    return parser.parse_args()


def configure_style() -> None:
    """Configure a restrained journal-ready Seaborn theme."""

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def load_tables(input_dir: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(input_dir / "animal_phase_summary.csv")
    bins = pd.read_csv(input_dir / "time_bins_30s.csv")
    statistics = pd.read_csv(input_dir / "statistics.csv")
    summary["phase"] = pd.Categorical(summary["phase"], PHASE_ORDER, ordered=True)
    summary["genotype"] = pd.Categorical(summary["genotype"], GENOTYPE_ORDER, ordered=True)
    return summary, bins, statistics


def p_value(statistics: pd.DataFrame, test: str, metric: str, genotype: str | None = None) -> float:
    mask = (statistics["test"] == test) & (statistics["metric"] == metric)
    if genotype is not None:
        mask &= statistics["genotype"] == genotype
    rows = statistics[mask]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0]["p"])


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return r"$P$ = n.a."
    if value < 0.001:
        return r"$P$ < 0.001"
    return rf"$P$ = {value:.3f}"


def add_panel_label(ax: Axes, label: str, x: float = -0.16, y: float = 1.045) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def significance_bracket(
    ax: Axes,
    x1: float,
    x2: float,
    y: float,
    text: str,
    height_fraction: float = 0.035,
) -> None:
    """Draw a compact significance bracket in data coordinates."""

    transform = ax.get_yaxis_transform(which="grid")
    ymin, ymax = ax.get_ylim()
    if ax.get_yscale() == "linear":
        height = (ymax - ymin) * height_fraction
    else:
        height = max(abs(y) * 0.12, 0.05)
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="0.2", lw=0.75, clip_on=False)
    ax.annotate(text, ((x1 + x2) / 2, y + height), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=6.6)


def paired_boxstrip(
    ax: Axes,
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    statistics: pd.DataFrame,
    linthresh: float,
    yticks: list[float],
    ylim: tuple[float, float],
    bracket_levels: tuple[float, float],
) -> None:
    """Plot paired animals as lines behind Seaborn box and strip layers."""

    offsets = {"WT": -0.18, "HET": 0.18}
    for genotype in GENOTYPE_ORDER:
        subset = data[data["genotype"] == genotype]
        wide = subset.pivot(index="animal_id", columns="phase", values=metric)
        for _, row in wide.iterrows():
            ax.plot(
                [offsets[genotype], 1 + offsets[genotype]],
                [row["alone"], row["partner"]],
                color=PALETTE[genotype],
                alpha=0.25,
                lw=0.65,
                zorder=1,
            )

    sns.boxplot(
        data=data,
        x="phase",
        y=metric,
        hue="genotype",
        order=PHASE_ORDER,
        hue_order=GENOTYPE_ORDER,
        palette=PALETTE,
        width=0.58,
        dodge=True,
        showfliers=False,
        saturation=0.75,
        linewidth=0.8,
        boxprops={"alpha": 0.28, "zorder": 2},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        medianprops={"color": "0.15", "linewidth": 1.1},
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="phase",
        y=metric,
        hue="genotype",
        order=PHASE_ORDER,
        hue_order=GENOTYPE_ORDER,
        palette=PALETTE,
        dodge=True,
        jitter=0.055,
        size=3.2,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.95,
        zorder=3,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_yscale("symlog", linthresh=linthresh, linscale=0.8)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{tick:g}" for tick in yticks])
    ax.set_xticks([0, 1], PHASE_LABELS)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.90", lw=0.55)
    ax.grid(axis="x", visible=False)

    wt_p = p_value(statistics, "paired Wilcoxon partner vs alone", metric, "WT")
    het_p = p_value(statistics, "paired Wilcoxon partner vs alone", metric, "HET")
    significance_bracket(ax, offsets["WT"], 1 + offsets["WT"], bracket_levels[0], f"WT: {format_p(wt_p)}")
    significance_bracket(ax, offsets["HET"], 1 + offsets["HET"], bracket_levels[1], f"Het: {format_p(het_p)}")


def change_scores(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    wide = data.pivot(index=["animal_id", "genotype"], columns="phase", values=metric).reset_index()
    wide["change"] = wide["partner"] - wide["alone"]
    return wide


def change_boxstrip(
    ax: Axes,
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    statistics: pd.DataFrame,
    linthresh: float,
    yticks: list[float],
    ylim: tuple[float, float],
    bracket_y: float,
) -> None:
    changes = change_scores(data, metric)
    sns.boxplot(
        data=changes,
        x="genotype",
        y="change",
        order=GENOTYPE_ORDER,
        hue="genotype",
        palette=PALETTE,
        legend=False,
        width=0.5,
        showfliers=False,
        saturation=0.75,
        linewidth=0.8,
        boxprops={"alpha": 0.28},
        medianprops={"color": "0.15", "linewidth": 1.1},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        ax=ax,
    )
    sns.stripplot(
        data=changes,
        x="genotype",
        y="change",
        order=GENOTYPE_ORDER,
        hue="genotype",
        palette=PALETTE,
        legend=False,
        jitter=0.11,
        size=3.5,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.95,
        ax=ax,
    )
    ax.axhline(0, color="0.45", lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_yscale("symlog", linthresh=linthresh, linscale=0.8)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{tick:g}" for tick in yticks])
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0, 1], ["WT", "Het"])
    ax.grid(axis="y", color="0.90", lw=0.55)
    ax.grid(axis="x", visible=False)
    interaction_p = p_value(statistics, "Mann-Whitney WT vs HET change", metric, "WT_vs_HET")
    significance_bracket(ax, 0, 1, bracket_y, format_p(interaction_p))


def timeline(ax: Axes, bins: pd.DataFrame) -> None:
    """Plot median and IQR animal-level call counts in each 30-second bin."""

    ax.axvspan(300, 600, color="0.965", zorder=0)
    for genotype in GENOTYPE_ORDER:
        subset = bins[bins["genotype"] == genotype]
        grouped = subset.groupby("bin_mid_s")["call_count"]
        x = grouped.median().index.to_numpy(float)
        median = grouped.median().to_numpy(float)
        q25 = grouped.quantile(0.25).to_numpy(float)
        q75 = grouped.quantile(0.75).to_numpy(float)
        color = PALETTE[genotype]
        ax.fill_between(x, q25, q75, color=color, alpha=0.14, linewidth=0)
        ax.plot(x, median, color=color, lw=1.4, marker="o", ms=2.2, label="WT" if genotype == "WT" else "Het")
    ax.axvline(300, color="0.35", lw=0.8, ls=(0, (3, 2)))
    ax.text(306, 0.97, "+ partner", transform=ax.get_xaxis_transform(), va="top", color="0.35", fontsize=7)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 36)
    ax.set_xticks(np.arange(0, 601, 100))
    ax.set_xlabel("Time in session (s)")
    ax.set_ylabel("Calls per 30 s, median (IQR)")
    ax.grid(axis="y", color="0.90", lw=0.55)
    ax.grid(axis="x", visible=False)
    if ax.legend_ is not None:
        ax.legend_.remove()


def make_figure(summary: pd.DataFrame, bins: pd.DataFrame, statistics: pd.DataFrame) -> plt.Figure:
    configure_style()
    figure = plt.figure(figsize=(7.2, 6.65), constrained_layout=False)
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.0, 0.74],
        left=0.09,
        right=0.985,
        bottom=0.08,
        top=0.94,
        wspace=0.34,
        hspace=0.36,
    )

    ax_a = figure.add_subplot(grid[0, 0])
    paired_boxstrip(
        ax_a,
        summary,
        "call_rate_per_min",
        r"Calls min$^{-1}$",
        statistics,
        linthresh=1.0,
        yticks=[0, 1, 10, 100],
        ylim=(0, 360),
        bracket_levels=(165, 270),
    )
    add_panel_label(ax_a, "a")

    ax_b = figure.add_subplot(grid[0, 1])
    change_boxstrip(
        ax_b,
        summary,
        "call_rate_per_min",
        r"Change in calls min$^{-1}$",
        statistics,
        linthresh=1.0,
        yticks=[-1, 0, 1, 10, 100],
        ylim=(-2.5, 360),
        bracket_y=180,
    )
    add_panel_label(ax_b, "b")

    ax_c = figure.add_subplot(grid[1, 0])
    paired_boxstrip(
        ax_c,
        summary,
        "vocal_output_s_per_min",
        r"Vocal output (s min$^{-1}$)",
        statistics,
        linthresh=0.002,
        yticks=[0, 0.001, 0.01, 0.1, 1],
        ylim=(0, 6),
        bracket_levels=(2.5, 4.4),
    )
    add_panel_label(ax_c, "c")

    ax_d = figure.add_subplot(grid[1, 1])
    change_boxstrip(
        ax_d,
        summary,
        "vocal_output_s_per_min",
        r"Change in vocal output (s min$^{-1}$)",
        statistics,
        linthresh=0.005,
        yticks=[-0.01, 0, 0.01, 0.1, 1],
        ylim=(-0.02, 6),
        bracket_y=2.5,
    )
    add_panel_label(ax_d, "d")

    ax_e = figure.add_subplot(grid[2, :])
    timeline(ax_e, bins)
    add_panel_label(ax_e, "e", x=-0.065, y=1.16)

    handles = [
        Line2D([0], [0], marker="o", color=WT_COLOR, lw=1.2, markersize=4, label="WT (n = 12)"),
        Line2D([0], [0], marker="o", color=HET_COLOR, lw=1.2, markersize=4, label="Het (n = 12)"),
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.995), ncol=2, frameon=False, handlelength=1.8)
    sns.despine(fig=figure)
    return figure


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_stem = args.output_stem.resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    summary, bins, statistics = load_tables(input_dir)
    figure = make_figure(summary, bins, statistics)
    for suffix, dpi in [(".png", 600), (".pdf", 300), (".svg", 300)]:
        figure.savefig(output_stem.with_suffix(suffix), dpi=dpi, facecolor="white")
    plt.close(figure)
    print(f"wrote: {output_stem}.png")
    print(f"wrote: {output_stem}.pdf")
    print(f"wrote: {output_stem}.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
