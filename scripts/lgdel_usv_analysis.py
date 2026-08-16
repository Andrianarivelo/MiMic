"""Analyze LgDel close-loop USV detections as WT versus HET.

Inputs are the upstream MiMic/VocalPy ``*_stats.csv`` files created next to each
WAV. The requested design is encoded directly:

* alone: 0 to 300 s
* partner: 300 to 600 s (duration-matched to the 5 min alone baseline)
* genotype comparison: WT versus HET, collapsed across virus

The script writes clean CSV tables and a multi-panel figure similar to the
reference figure supplied by the user.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    from matplotlib.transforms import blended_transform_factory
except ModuleNotFoundError:  # Tables and statistics can run without plotting extras.
    plt = None
    Axes = Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE_ORDER = ["alone", "partner"]
PHASE_LABEL = {"alone": "alone", "partner": "+ partner"}
ANALYSIS_END_S = 600.0
PHASE_LIMITS = {"alone": (0.0, 300.0), "partner": (300.0, ANALYSIS_END_S)}
GENOTYPE_ORDER = ["WT", "HET"]
GENOTYPE_LABEL = {"WT": "WT", "HET": "Het"}
GENOTYPE_COLOR = {"WT": "#1f77b4", "HET": "#ff7f0e"}
SYLLABLE_CLASSES = [
    "chevron",
    "complex",
    "down_fm",
    "flat",
    "mult_steps",
    "rev_chevron",
    "short",
    "step_down",
    "step_up",
    "two_steps",
    "up_fm",
]
PCA_FEATURES = [
    "duration(ms)",
    "interval(s)",
    "min_freq",
    "max_freq",
    "avg_freq",
    "bandwidth",
    "min_intensity",
    "max_intensity",
    "avg_intensity",
    "bg_intensity",
    "area(pixels)",
    "centroid_y",
]
PCA_FEATURE_LABELS = {
    "duration(ms)": "Duration",
    "interval(s)": "Interval",
    "min_freq": "Min frequency",
    "max_freq": "Max frequency",
    "avg_freq": "Mean frequency",
    "bandwidth": "Bandwidth",
    "min_intensity": "Min intensity",
    "max_intensity": "Max intensity",
    "avg_intensity": "Mean intensity",
    "bg_intensity": "Background intensity",
    "area(pixels)": "Area",
    "centroid_y": "Spectral centroid",
}
CLASS_COLORS = {
    "chevron": "#38bdf8", "complex": "#a78bfa", "down_fm": "#f472b6",
    "flat": "#34d399", "mult_steps": "#fbbf24", "rev_chevron": "#22d3ee",
    "short": "#fb7185", "step_down": "#818cf8", "step_up": "#2dd4bf",
    "two_steps": "#f59e0b", "up_fm": "#4ade80",
}


@dataclass(frozen=True)
class CohortEntry:
    """One annotated animal and its expected detector CSV."""

    animal_id: str
    genotype: str
    virus: str
    wav_path: pathlib.Path
    stats_path: pathlib.Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse paths for the local LgDel analysis."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=pathlib.Path, default=ROOT.parent)
    parser.add_argument("--trial-plan", type=pathlib.Path, default=ROOT.parent / "pykaboo_trial_plan.csv")
    parser.add_argument(
        "--completed-plan",
        type=pathlib.Path,
        default=ROOT.parent / "pykaboo_trial_plan_completed.xlsx",
    )
    parser.add_argument(
        "--cohort-manifest",
        type=pathlib.Path,
        help="Optional validated manifest CSV to use instead of rebuilding cohort labels from trial plans.",
    )
    parser.add_argument(
        "--cohort-labels",
        type=pathlib.Path,
        help="Optional CSV with animal_id/genotype labels that override labels in --cohort-manifest.",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "lgdel_usv_analysis")
    parser.add_argument("--bin-seconds", type=float, default=30.0)
    parser.add_argument(
        "--partner-window-seconds",
        type=float,
        default=300.0,
        help="Seconds after partner introduction to include (default: 300 for a duration-matched window).",
    )
    parser.add_argument("--no-figures", action="store_true", help="Write tables and report without Matplotlib figures.")
    return parser.parse_args(argv)


def normalize_genotype(value: object) -> str:
    """Normalize local genotype labels to WT, HET, or UNKNOWN."""

    text = "" if value is None else str(value).strip().upper()
    if text.startswith("WT"):
        return "WT"
    if text.startswith("HET") or text in {"HE", "HT"}:
        return "HET"
    return "UNKNOWN"


def clean_string(value: object) -> str:
    """Return a clean string for metadata cells."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def p_text(p_value: float) -> str:
    """Format a p value for compact figure annotations."""

    if not np.isfinite(p_value):
        return "p=n/a"
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}"


def p_stars(p_value: float) -> str:
    """Return standard significance stars."""

    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def sem(values: np.ndarray) -> float:
    """Compute SEM, returning zero for single-animal groups."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def phase_of(time_s: float) -> str:
    """Assign an absolute call time to the requested behavioral phase."""

    if time_s < 300.0:
        return "alone"
    if time_s < ANALYSIS_END_S:
        return "partner"
    return "outside"


def load_cohort(data_root: pathlib.Path, trial_plan: pathlib.Path, completed_plan: pathlib.Path) -> list[CohortEntry]:
    """Load trial annotations, preferring completed genotype and virus labels."""

    plan = pd.read_csv(trial_plan)
    try:
        completed = pd.read_excel(completed_plan)
    except ImportError:
        completed = pd.DataFrame(columns=["Animal ID", "genotype_pyrat", "virus"])
    plan["animal_id"] = plan["Animal ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    completed["animal_id"] = completed["Animal ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    keep = [c for c in ["animal_id", "genotype_pyrat", "virus"] if c in completed.columns]
    merged = plan.merge(completed[keep], on="animal_id", how="left")

    entries: list[CohortEntry] = []
    for _, row in merged.iterrows():
        animal_id = clean_string(row["animal_id"])
        if not animal_id.isdigit():
            continue
        genotype = normalize_genotype(row.get("genotype_pyrat", row.get("genotype")))
        if genotype not in GENOTYPE_ORDER:
            genotype = normalize_genotype(row.get("genotype"))
        stem = f"{animal_id}_1_baseline"
        wav_path = data_root / animal_id / "baseline" / "1" / f"{stem}.wav"
        stats_path = data_root / animal_id / "baseline" / "1" / f"{stem}_outputs" / f"{stem}_stats.csv"
        entries.append(
            CohortEntry(
                animal_id=animal_id,
                genotype=genotype,
                virus=clean_string(row.get("virus")),
                wav_path=wav_path,
                stats_path=stats_path,
            )
        )
    return entries


def load_cohort_manifest(path: pathlib.Path, labels_path: pathlib.Path | None = None) -> list[CohortEntry]:
    """Load exact cohort labels and paths from a previously validated manifest."""

    manifest = pd.read_csv(path, dtype={"animal_id": str})
    if labels_path is not None:
        labels = pd.read_csv(labels_path, dtype={"animal_id": str})
        keep = [column for column in ["animal_id", "genotype", "virus"] if column in labels.columns]
        labels = labels[keep].drop_duplicates(subset="animal_id")
        manifest = manifest.drop(columns=[column for column in ["genotype", "virus"] if column in manifest.columns])
        manifest = manifest.merge(labels, on="animal_id", how="left", validate="one_to_one")
    entries = []
    for _, row in manifest.iterrows():
        entries.append(
            CohortEntry(
                animal_id=clean_string(row["animal_id"]),
                genotype=normalize_genotype(row["genotype"]),
                virus=clean_string(row.get("virus")),
                wav_path=pathlib.Path(clean_string(row["wav_path"])),
                stats_path=pathlib.Path(clean_string(row["stats_path"])),
            )
        )
    return entries


def load_one_stats(entry: CohortEntry) -> pd.DataFrame:
    """Load one upstream stats CSV and attach animal metadata."""

    if not entry.stats_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(entry.stats_path)
    unnamed = [col for col in frame.columns if str(col).startswith("Unnamed") or str(col).startswith("H")]
    frame = frame.drop(columns=unnamed, errors="ignore")
    if "start(s)" not in frame.columns:
        raise ValueError(f"{entry.stats_path} is missing start(s)")
    frame["animal_id"] = entry.animal_id
    frame["genotype"] = entry.genotype
    frame["virus"] = entry.virus
    frame["phase"] = frame["start(s)"].astype(float).map(phase_of)
    frame = frame[frame["phase"].isin(PHASE_ORDER)].copy()
    frame["phase"] = pd.Categorical(frame["phase"], categories=PHASE_ORDER, ordered=True)
    frame["duration_s"] = frame["duration(ms)"].astype(float) / 1000.0
    frame["avg_freq_khz"] = frame["avg_freq"].astype(float) / 1000.0
    frame["bandwidth_khz"] = frame["bandwidth"].astype(float) / 1000.0
    frame["class_top1"] = frame["class_top1"].fillna("unknown").astype(str)
    return frame


def load_calls(entries: list[CohortEntry]) -> pd.DataFrame:
    """Concatenate all available detector CSVs."""

    frames = [load_one_stats(entry) for entry in entries]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_manifest(entries: list[CohortEntry]) -> pd.DataFrame:
    """Build one manifest row per expected animal."""

    rows = []
    for entry in entries:
        rows.append(
            {
                "animal_id": entry.animal_id,
                "genotype": entry.genotype,
                "virus": entry.virus,
                "wav_path": str(entry.wav_path),
                "stats_path": str(entry.stats_path),
                "wav_exists": entry.wav_path.exists(),
                "stats_exists": entry.stats_path.exists(),
            }
        )
    return pd.DataFrame(rows)


def build_phase_summary(calls: pd.DataFrame, entries: list[CohortEntry]) -> pd.DataFrame:
    """Aggregate calls into animal-level phase outcomes."""

    rows = []
    for entry in entries:
        animal_calls = calls[calls["animal_id"] == entry.animal_id] if not calls.empty else pd.DataFrame()
        for phase in PHASE_ORDER:
            start_s, end_s = PHASE_LIMITS[phase]
            phase_min = (end_s - start_s) / 60.0
            sub = animal_calls[animal_calls["phase"] == phase] if not animal_calls.empty else pd.DataFrame()
            call_count = int(len(sub))
            total_duration_s = float(sub["duration_s"].sum()) if call_count else 0.0
            rows.append(
                {
                    "animal_id": entry.animal_id,
                    "genotype": entry.genotype,
                    "virus": entry.virus,
                    "phase": phase,
                    "phase_min": phase_min,
                    "call_count": call_count,
                    "call_rate_per_min": call_count / phase_min,
                    "total_call_duration_s": total_duration_s,
                    "vocal_output_s_per_min": total_duration_s / phase_min,
                    "mean_duration_ms": float(sub["duration(ms)"].mean()) if call_count else np.nan,
                    "mean_avg_freq_khz": float(sub["avg_freq_khz"].mean()) if call_count else np.nan,
                    "mean_bandwidth_khz": float(sub["bandwidth_khz"].mean()) if call_count else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out["phase"] = pd.Categorical(out["phase"], categories=PHASE_ORDER, ordered=True)
    return out


def build_time_bins(calls: pd.DataFrame, entries: list[CohortEntry], bin_seconds: float) -> pd.DataFrame:
    """Build animal-level call-count timelines in fixed-width bins."""

    edges = np.arange(0.0, ANALYSIS_END_S + bin_seconds, bin_seconds)
    rows = []
    for entry in entries:
        sub = calls[calls["animal_id"] == entry.animal_id] if not calls.empty else pd.DataFrame()
        starts = sub["start(s)"].to_numpy(dtype=float) if not sub.empty else np.array([])
        counts, _ = np.histogram(starts, bins=edges)
        for index, count in enumerate(counts):
            rows.append(
                {
                    "animal_id": entry.animal_id,
                    "genotype": entry.genotype,
                    "virus": entry.virus,
                    "bin_start_s": edges[index],
                    "bin_end_s": edges[index + 1],
                    "bin_mid_s": 0.5 * (edges[index] + edges[index + 1]),
                    "phase": phase_of(edges[index]),
                    "call_count": int(count),
                }
            )
    return pd.DataFrame(rows)


def mann_whitney_table(phase_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare WT versus HET per phase for main animal-level metrics."""

    metrics = [
        "call_count",
        "call_rate_per_min",
        "total_call_duration_s",
        "vocal_output_s_per_min",
        "mean_duration_ms",
        "mean_avg_freq_khz",
        "mean_bandwidth_khz",
    ]
    rows = []
    for phase in PHASE_ORDER:
        sub = phase_summary[phase_summary["phase"] == phase]
        for metric in metrics:
            wt = sub.loc[sub["genotype"] == "WT", metric].to_numpy(dtype=float)
            het = sub.loc[sub["genotype"] == "HET", metric].to_numpy(dtype=float)
            wt = wt[np.isfinite(wt)]
            het = het[np.isfinite(het)]
            if len(wt) and len(het):
                test = stats.mannwhitneyu(wt, het, alternative="two-sided")
                p_value = float(test.pvalue)
                u_value = float(test.statistic)
                effect = 2.0 * u_value / (len(wt) * len(het)) - 1.0
            else:
                p_value = np.nan
                u_value = np.nan
                effect = np.nan
            rows.append(
                {
                    "test": "Mann-Whitney WT vs HET",
                    "phase": phase,
                    "metric": metric,
                    "n_WT": int(len(wt)),
                    "n_HET": int(len(het)),
                    "mean_WT": float(np.mean(wt)) if len(wt) else np.nan,
                    "mean_HET": float(np.mean(het)) if len(het) else np.nan,
                    "U": u_value,
                    "p": p_value,
                    "rank_biserial": effect,
                }
            )
    return pd.DataFrame(rows)


def paired_change_table(phase_summary: pd.DataFrame) -> pd.DataFrame:
    """Test paired pre/post changes and compare change scores by genotype."""

    metrics = [
        "call_count",
        "call_rate_per_min",
        "total_call_duration_s",
        "vocal_output_s_per_min",
        "mean_duration_ms",
        "mean_avg_freq_khz",
        "mean_bandwidth_khz",
    ]
    rows = []
    for metric in metrics:
        wide = phase_summary.pivot(index=["animal_id", "genotype"], columns="phase", values=metric).reset_index()
        wide["change"] = wide["partner"] - wide["alone"]
        for genotype in GENOTYPE_ORDER:
            sub = wide[wide["genotype"] == genotype].dropna(subset=["alone", "partner"])
            pre = sub["alone"].to_numpy(dtype=float)
            post = sub["partner"].to_numpy(dtype=float)
            if len(pre):
                try:
                    test = stats.wilcoxon(post, pre, alternative="two-sided")
                    statistic = float(test.statistic)
                    p_value = float(test.pvalue)
                except ValueError:
                    statistic = 0.0
                    p_value = 1.0
            else:
                statistic = np.nan
                p_value = np.nan
            rows.append(
                {
                    "test": "paired Wilcoxon partner vs alone",
                    "phase": "partner_minus_alone",
                    "genotype": genotype,
                    "metric": metric,
                    "n": int(len(pre)),
                    "mean_alone": float(np.mean(pre)) if len(pre) else np.nan,
                    "mean_partner": float(np.mean(post)) if len(post) else np.nan,
                    "mean_change": float(np.mean(post - pre)) if len(pre) else np.nan,
                    "W": statistic,
                    "p": p_value,
                }
            )

        wt_change = wide.loc[wide["genotype"] == "WT", "change"].dropna().to_numpy(dtype=float)
        het_change = wide.loc[wide["genotype"] == "HET", "change"].dropna().to_numpy(dtype=float)
        if len(wt_change) and len(het_change):
            test = stats.mannwhitneyu(wt_change, het_change, alternative="two-sided")
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
            effect = 2.0 * statistic / (len(wt_change) * len(het_change)) - 1.0
        else:
            statistic = np.nan
            p_value = np.nan
            effect = np.nan
        rows.append(
            {
                "test": "Mann-Whitney WT vs HET change",
                "phase": "partner_minus_alone",
                "genotype": "WT_vs_HET",
                "metric": metric,
                "n_WT": int(len(wt_change)),
                "n_HET": int(len(het_change)),
                "mean_change_WT": float(np.mean(wt_change)) if len(wt_change) else np.nan,
                "mean_change_HET": float(np.mean(het_change)) if len(het_change) else np.nan,
                "U": statistic,
                "p": p_value,
                "rank_biserial": effect,
            }
        )
    return pd.DataFrame(rows)


def repertoire_stats(calls: pd.DataFrame) -> pd.DataFrame:
    """Run chi-square tests on class distributions by phase."""

    rows = []
    for phase in PHASE_ORDER:
        sub = calls[calls["phase"] == phase] if not calls.empty else pd.DataFrame()
        table = pd.crosstab(sub["class_top1"], sub["genotype"]) if not sub.empty else pd.DataFrame()
        table = table.reindex(index=SYLLABLE_CLASSES, columns=GENOTYPE_ORDER, fill_value=0)
        table = table.loc[table.sum(axis=1) > 0]
        if table.empty or table.shape[0] < 2 or (table.sum(axis=0) == 0).any():
            chi2 = np.nan
            p_value = np.nan
            dof = np.nan
        else:
            chi2, p_value, dof, _ = chi2_contingency(table.to_numpy())
        rows.append(
            {
                "test": "chi-square class distribution WT vs HET",
                "phase": phase,
                "metric": "class_top1",
                "n_WT": int(table["WT"].sum()) if "WT" in table else 0,
                "n_HET": int(table["HET"].sum()) if "HET" in table else 0,
                "chi2": chi2,
                "dof": dof,
                "p": p_value,
            }
        )
    return pd.DataFrame(rows)


def lookup_p(stats_frame: pd.DataFrame, phase: str, metric: str) -> float:
    """Find one p value from the statistics table."""

    sub = stats_frame[(stats_frame["phase"] == phase) & (stats_frame["metric"] == metric)]
    if sub.empty:
        return np.nan
    return float(sub.iloc[0]["p"])


def annotate_bracket(ax: Axes, x1: float, x2: float, y: float, p_value: float) -> None:
    """Draw a significance bracket on an axis."""

    if not np.isfinite(y):
        return
    height = max((ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.025, 0.02)
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="#333333", lw=1.0)
    ax.text((x1 + x2) / 2.0, y + height, f"{p_stars(p_value)}\n{p_text(p_value)}", ha="center", va="bottom", fontsize=8)


def annotate_bracket_fraction(ax: Axes, x1: float, x2: float, y: float, p_value: float) -> None:
    """Draw a compact bracket using an axis-fraction y position."""

    transform = blended_transform_factory(ax.transData, ax.transAxes)
    height = 0.025
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + height, y + height, y],
        color="#333333",
        lw=0.85,
        transform=transform,
        clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2.0,
        y + height + 0.01,
        f"{p_stars(p_value)}  {p_text(p_value)}",
        transform=transform,
        ha="center",
        va="bottom",
        fontsize=7.2,
    )


def apply_style() -> None:
    """Apply a clean publication-style Matplotlib theme."""

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": "#e8e8e8",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_repertoire(ax: Axes, calls: pd.DataFrame, stats_frame: pd.DataFrame) -> None:
    """Plot class composition as stacked proportions."""

    cols = [(g, p) for p in PHASE_ORDER for g in GENOTYPE_ORDER]
    x = np.arange(len(cols))
    bottom = np.zeros(len(cols), dtype=float)
    for cls in SYLLABLE_CLASSES:
        values = []
        for genotype, phase in cols:
            sub = calls[(calls["genotype"] == genotype) & (calls["phase"] == phase)] if not calls.empty else pd.DataFrame()
            denom = max(len(sub), 1)
            values.append(float((sub["class_top1"] == cls).sum()) / denom if not sub.empty else 0.0)
        color = CLASS_COLORS.get(cls, "#777777")
        ax.bar(x, values, bottom=bottom, width=0.74, color=color, edgecolor="white", linewidth=0.35, label=cls)
        bottom += np.asarray(values)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{GENOTYPE_LABEL[g]}\n{PHASE_LABEL[p]}" for g, p in cols])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("within-column proportion")
    ax.set_title("Syllable repertoire composition", pad=17)
    for xpos, (genotype, phase) in zip(x, cols):
        n_calls = len(calls[(calls["genotype"] == genotype) & (calls["phase"] == phase)]) if not calls.empty else 0
        ax.text(xpos, 1.02, f"n={n_calls}", ha="center", va="bottom", fontsize=8)
    for phase, pair in {"alone": (0, 1), "partner": (2, 3)}.items():
        annotate_bracket(ax, pair[0], pair[1], 1.08, lookup_p(stats_frame, phase, "class_top1"))
    ax.set_xlim(-0.55, 5.45)
    ax.legend(
        title="class",
        fontsize=5.7,
        title_fontsize=6.5,
        loc="center right",
        bbox_to_anchor=(1.0, 0.54),
        handlelength=1.2,
        labelspacing=0.28,
        borderaxespad=0.15,
    )
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.6)


def plot_total_calls(ax: Axes, phase_summary: pd.DataFrame) -> None:
    """Plot total call counts per animal."""

    totals = phase_summary.groupby(["animal_id", "genotype"], as_index=False)["call_count"].sum()
    totals = totals.sort_values(["genotype", "call_count"], ascending=[True, False])
    x = np.arange(len(totals))
    colors = [GENOTYPE_COLOR[g] for g in totals["genotype"]]
    ax.bar(x, totals["call_count"], color=colors, edgecolor="white", linewidth=0.6)
    ymax = max(float(totals["call_count"].max()), 1.0)
    for xpos, value in zip(x, totals["call_count"]):
        ax.text(xpos, value + ymax * 0.015, str(int(value)), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(totals["animal_id"], rotation=75, ha="right", rotation_mode="anchor", fontsize=6.2)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylabel("Total calls")
    ax.set_title("Total vocal output per mouse", pad=8)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.6)


def plot_call_rate(ax: Axes, phase_summary: pd.DataFrame, stats_frame: pd.DataFrame) -> None:
    """Plot animal-level call rate as compact Seaborn box-and-strip groups."""

    plot_data = phase_summary.copy()
    plot_data["phase"] = plot_data["phase"].astype(str)
    plot_data["genotype"] = plot_data["genotype"].astype(str)
    sns.boxplot(
        data=plot_data,
        x="phase",
        y="call_rate_per_min",
        hue="genotype",
        order=PHASE_ORDER,
        hue_order=GENOTYPE_ORDER,
        palette=GENOTYPE_COLOR,
        width=0.62,
        gap=0.16,
        showfliers=False,
        saturation=0.72,
        linewidth=0.85,
        boxprops={"alpha": 0.38},
        medianprops={"color": "#222222", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.85},
        capprops={"linewidth": 0.85},
        ax=ax,
    )
    sns.stripplot(
        data=plot_data,
        x="phase",
        y="call_rate_per_min",
        hue="genotype",
        order=PHASE_ORDER,
        hue_order=GENOTYPE_ORDER,
        palette=GENOTYPE_COLOR,
        dodge=True,
        jitter=0.075,
        size=4.0,
        alpha=0.92,
        edgecolor="white",
        linewidth=0.45,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.85)
    ax.set_ylim(0, 210)
    ax.set_yticks([0, 1, 10, 100])
    ax.set_yticklabels(["0", "1", "10", "100"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([PHASE_LABEL[p] for p in PHASE_ORDER])
    ax.set_xlabel("")
    ax.set_ylabel("Calls / min")
    ax.set_title("Call rate per genotype × phase", pad=8)
    for index, phase in enumerate(PHASE_ORDER):
        annotate_bracket_fraction(
            ax,
            index - 0.20,
            index + 0.20,
            0.78 if phase == "alone" else 0.89,
            lookup_p(stats_frame, phase, "call_rate_per_min"),
        )
    handles = [
        Line2D([0], [0], marker="o", color=GENOTYPE_COLOR[genotype], markerfacecolor=GENOTYPE_COLOR[genotype], lw=1.2, markersize=4.5,
               label=f"{GENOTYPE_LABEL[genotype]} (n={plot_data[plot_data['genotype'] == genotype]['animal_id'].nunique()})")
        for genotype in GENOTYPE_ORDER
    ]
    ax.legend(handles=handles, loc="upper left", ncol=2, handlelength=1.4, columnspacing=0.9)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.6)


def plot_timeline(ax: Axes, time_bins: pd.DataFrame) -> None:
    """Plot mean call counts per 30 s bin."""

    for genotype in GENOTYPE_ORDER:
        sub = time_bins[time_bins["genotype"] == genotype]
        grouped = sub.groupby("bin_mid_s")["call_count"]
        x = grouped.mean().index.to_numpy(dtype=float)
        mean = grouped.mean().to_numpy(dtype=float)
        err = grouped.apply(lambda s: sem(s.to_numpy(dtype=float))).to_numpy(dtype=float)
        ax.plot(x, mean, color=GENOTYPE_COLOR[genotype], lw=1.7, label=GENOTYPE_LABEL[genotype])
        ax.fill_between(x, mean - err, mean + err, color=GENOTYPE_COLOR[genotype], alpha=0.20, linewidth=0)
    ax.axvline(300, color="#555555", ls="--", lw=1.0)
    ax.text(306, ax.get_ylim()[1] * 0.92, "+ partner", color="#555555", fontsize=9)
    ax.set_xlim(0, ANALYSIS_END_S)
    ax.set_xlabel("Time in session (s)")
    ax.set_ylabel("Calls / 30 s bin")
    ax.set_title("Call timeline: partner introduction at 5 min", pad=8)
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.6)


def plot_vocal_output(ax: Axes, phase_summary: pd.DataFrame, stats_frame: pd.DataFrame) -> None:
    """Plot call-duration output per minute by social phase."""

    x = np.arange(len(PHASE_ORDER))
    for genotype in GENOTYPE_ORDER:
        sub = phase_summary[phase_summary["genotype"] == genotype]
        means = []
        errors = []
        for phase in PHASE_ORDER:
            values = sub.loc[sub["phase"] == phase, "vocal_output_s_per_min"].to_numpy(dtype=float)
            means.append(float(np.mean(values)) if len(values) else np.nan)
            errors.append(sem(values))
        ax.errorbar(x, means, yerr=errors, color=GENOTYPE_COLOR[genotype], marker="o", lw=2.0, capsize=4, label=GENOTYPE_LABEL[genotype])
        for _, animal in sub.groupby("animal_id"):
            y = [float(animal.loc[animal["phase"] == phase, "vocal_output_s_per_min"].iloc[0]) for phase in PHASE_ORDER]
            ax.plot(x, y, color=GENOTYPE_COLOR[genotype], alpha=0.18, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABEL[p] for p in PHASE_ORDER])
    ax.set_ylabel("Call duration (s / min)")
    ax.set_title("Vocal output by social phase", pad=8)
    ymax = ax.get_ylim()[1]
    for index, phase in enumerate(PHASE_ORDER):
        annotate_bracket(ax, index - 0.10, index + 0.10, ymax * (0.78 + index * 0.10), lookup_p(stats_frame, phase, "vocal_output_s_per_min"))
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.50))
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.6)


def plot_acoustics(ax: Axes, calls: pd.DataFrame) -> None:
    """Plot per-animal acoustic feature means."""

    if calls.empty:
        ax.text(0.5, 0.5, "No calls", transform=ax.transAxes, ha="center", va="center")
        return
    features = calls.groupby(["animal_id", "genotype"], as_index=False).agg(
        mean_duration_ms=("duration(ms)", "mean"),
        mean_avg_freq_khz=("avg_freq_khz", "mean"),
    )
    for genotype in GENOTYPE_ORDER:
        sub = features[features["genotype"] == genotype]
        ax.scatter(
            sub["mean_duration_ms"],
            sub["mean_avg_freq_khz"],
            s=44,
            color=GENOTYPE_COLOR[genotype],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.88,
            label=GENOTYPE_LABEL[genotype],
        )
    ax.set_xlabel("Mean duration (ms)")
    ax.set_ylabel("Mean average frequency (kHz)")
    ax.set_title("Call acoustic features", pad=8)
    ax.legend(loc="best", fontsize=8)
    ax.grid(color="#e8e8e8", linewidth=0.6)


def build_pca_tables(calls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build an animal-level standardized PCA from nonredundant call features."""

    available = [feature for feature in PCA_FEATURES if feature in calls.columns]
    animal_features = (
        calls.groupby(["animal_id", "genotype"], as_index=False)[available]
        .median()
        .copy()
    )
    feature_data = animal_features[available].replace([np.inf, -np.inf], np.nan)
    feature_data = feature_data.fillna(feature_data.median())
    standard_deviation = feature_data.std(axis=0, ddof=0)
    available = standard_deviation[standard_deviation > 1e-12].index.tolist()
    standardized = (feature_data[available] - feature_data[available].mean(axis=0)) / feature_data[available].std(axis=0, ddof=0)
    u_matrix, singular_values, components = np.linalg.svd(standardized.to_numpy(dtype=float), full_matrices=False)
    scores_array = u_matrix * singular_values
    explained_ratio = singular_values**2 / np.sum(singular_values**2)

    # Orient PC1 so that positive scores point toward the WT centroid. PCA signs
    # are arbitrary, so this only stabilizes interpretation across reruns.
    wt_mask = animal_features["genotype"].astype(str).to_numpy() == "WT"
    het_mask = animal_features["genotype"].astype(str).to_numpy() == "HET"
    if scores_array[wt_mask, 0].mean() < scores_array[het_mask, 0].mean():
        scores_array[:, 0] *= -1.0
        components[0, :] *= -1.0

    score_columns = {f"PC{index + 1}": scores_array[:, index] for index in range(scores_array.shape[1])}
    scores = pd.concat(
        [animal_features[["animal_id", "genotype"]].reset_index(drop=True), pd.DataFrame(score_columns)],
        axis=1,
    )
    loadings = pd.DataFrame(
        {
            "feature": available,
            "feature_label": [PCA_FEATURE_LABELS.get(feature, feature) for feature in available],
            **{f"PC{index + 1}": components[index, :] for index in range(components.shape[0])},
        }
    )
    variance = pd.DataFrame(
        {
            "component": [f"PC{index + 1}" for index in range(len(explained_ratio))],
            "explained_variance_ratio": explained_ratio,
            "cumulative_explained_variance": np.cumsum(explained_ratio),
        }
    )
    return scores, loadings, variance


def add_data_ellipse(ax: Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    """Add a translucent 68% bivariate-normal data ellipse."""

    if len(x) < 3:
        return
    covariance = np.cov(np.column_stack([x, y]), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    radius = math.sqrt(float(stats.chi2.ppf(0.68, df=2)))
    ellipse = Ellipse(
        (float(np.mean(x)), float(np.mean(y))),
        width=2.0 * radius * math.sqrt(max(float(eigenvalues[0]), 0.0)),
        height=2.0 * radius * math.sqrt(max(float(eigenvalues[1]), 0.0)),
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=1.0,
        alpha=0.12,
        zorder=1,
    )
    ax.add_patch(ellipse)


def plot_pca(ax: Axes, calls: pd.DataFrame) -> None:
    """Plot animal PCA scores in 3D with explained variance and top loadings."""

    scores, loadings, variance = build_pca_tables(calls)
    for genotype in GENOTYPE_ORDER:
        subset = scores[scores["genotype"].astype(str) == genotype]
        x = subset["PC1"].to_numpy(dtype=float)
        y = subset["PC2"].to_numpy(dtype=float)
        z = subset["PC3"].to_numpy(dtype=float)
        ax.scatter(
            x,
            y,
            z,
            s=35,
            color=GENOTYPE_COLOR[genotype],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
            label=GENOTYPE_LABEL[genotype],
            depthshade=False,
            zorder=3,
        )
        ax.scatter(
            [float(np.mean(x))],
            [float(np.mean(y))],
            [float(np.mean(z))],
            marker="X",
            s=48,
            color=GENOTYPE_COLOR[genotype],
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
    pc1_variance = 100.0 * float(variance.iloc[0]["explained_variance_ratio"])
    pc2_variance = 100.0 * float(variance.iloc[1]["explained_variance_ratio"])
    pc3_variance = 100.0 * float(variance.iloc[2]["explained_variance_ratio"])
    cumulative = pc1_variance + pc2_variance + pc3_variance
    ax.set_xlabel(f"PC1 ({pc1_variance:.1f}%)", labelpad=-1, fontsize=6.5)
    ax.set_ylabel(f"PC2 ({pc2_variance:.1f}%)", labelpad=-1, fontsize=6.5)
    ax.set_zlabel(f"PC3 ({pc3_variance:.1f}%)", labelpad=-2, fontsize=6.5)
    ax.set_title(f"Animal-level 3D PCA of call features ({cumulative:.1f}% total)", pad=7)
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((1.25, 1.0, 0.95))
    ax.tick_params(axis="x", labelsize=5.5, pad=0)
    ax.tick_params(axis="y", labelsize=5.5, pad=0)
    ax.tick_params(axis="z", labelsize=5.5, pad=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.83), fontsize=6.3, borderaxespad=0.2)
    for axis_3d in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis_3d.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis_3d.pane.set_edgecolor("#d5d5d5")
        axis_3d._axinfo["grid"]["color"] = (0.90, 0.90, 0.90, 1.0)
        axis_3d._axinfo["grid"]["linewidth"] = 0.5
    top_labels = []
    for component in ["PC1", "PC2", "PC3"]:
        row = loadings.assign(magnitude=loadings[component].abs()).nlargest(1, "magnitude").iloc[0]
        top_labels.append(f"{component}: {row['feature_label']}")
    ax.text2D(
        0.99,
        0.98,
        "Top loadings  " + "  |  ".join(top_labels),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.4,
        color="#4a4a4a",
    )


def make_figure(calls: pd.DataFrame, phase_summary: pd.DataFrame, time_bins: pd.DataFrame, stats_frame: pd.DataFrame, output_dir: pathlib.Path) -> list[pathlib.Path]:
    """Create the requested multi-panel USV summary figure."""

    if plt is None:
        raise RuntimeError("Matplotlib is required to create figures. Install the project environment or pass --no-figures.")
    apply_style()
    fig = plt.figure(figsize=(14.2, 7.25), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.052,
        right=0.985,
        bottom=0.09,
        top=0.90,
        wspace=0.31,
        hspace=0.36,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0], projection="3d"),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    ]
    plot_repertoire(axes[0], calls, stats_frame)
    plot_total_calls(axes[1], phase_summary)
    plot_call_rate(axes[2], phase_summary, stats_frame)
    plot_pca(axes[3], calls)
    plot_timeline(axes[4], time_bins)
    plot_vocal_output(axes[5], phase_summary, stats_frame)
    for axis in [axes[0], axes[1], axes[2], axes[4], axes[5]]:
        sns.despine(ax=axis, offset=6, trim=True)
    plt.setp(
        axes[1].get_xticklabels(),
        rotation=75,
        ha="right",
        rotation_mode="anchor",
        fontsize=6.2,
    )
    fig.suptitle("LgDel USV analysis: WT vs HET, matched 5 min windows", fontsize=13, fontweight="bold", y=0.978)
    paths = [
        output_dir / "lgdel_usv_wt_vs_het_summary.png",
        output_dir / "lgdel_usv_wt_vs_het_summary.svg",
        output_dir / "lgdel_usv_wt_vs_het_summary.pdf",
    ]
    for path in paths:
        dpi = 600 if path.suffix.lower() == ".png" else 300
        fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return paths


def write_report(output_dir: pathlib.Path, manifest: pd.DataFrame, calls: pd.DataFrame, phase_summary: pd.DataFrame, stats_frame: pd.DataFrame, figures: list[pathlib.Path]) -> pathlib.Path:
    """Write a short Markdown report with key outputs and caveats."""

    report = output_dir / "lgdel_usv_analysis_report.md"
    n_by_genotype = phase_summary.groupby("genotype")["animal_id"].nunique().to_dict()
    missing = manifest[~manifest["stats_exists"]]
    key = stats_frame[(stats_frame["test"] == "Mann-Whitney WT vs HET") & (stats_frame["metric"].isin(["call_rate_per_min", "vocal_output_s_per_min"]))]
    lines = [
        "# LgDel USV WT vs HET Analysis",
        "",
        "## Design",
        "- Partner introduction: 300 s.",
        "- Alone phase: 0 to 300 s.",
        f"- Partner phase: 300 to {ANALYSIS_END_S:g} s ({(ANALYSIS_END_S - 300.0) / 60.0:g} min after introduction).",
        f"- Baseline duration: 5 min. Partner-window duration: {(ANALYSIS_END_S - 300.0) / 60.0:g} min.",
        "- Genotype grouping: WT vs HET, collapsed across virus.",
        "",
        "## Dataset",
        f"- Animals by genotype: `{json.dumps(n_by_genotype, sort_keys=True)}`",
        f"- Detected calls loaded: {len(calls)}",
        f"- Animals missing stats CSV: {len(missing)}",
        "",
        "## Main WT vs HET Tests",
    ]
    for _, row in key.iterrows():
        lines.append(
            f"- {row['metric']} in {row['phase']}: WT mean={row['mean_WT']:.4g}, HET mean={row['mean_HET']:.4g}, {p_text(float(row['p']))}, {p_stars(float(row['p']))}."
        )
    paired_key = stats_frame[
        (stats_frame["test"] == "paired Wilcoxon partner vs alone")
        & (stats_frame["metric"].isin(["call_rate_per_min", "vocal_output_s_per_min"]))
    ]
    lines.extend(["", "## Paired Pre/Post Tests Within Genotype"])
    for _, row in paired_key.iterrows():
        lines.append(
            f"- {row['metric']} in {row['genotype']}: pre mean={row['mean_alone']:.4g}, post mean={row['mean_partner']:.4g}, mean change={row['mean_change']:.4g}, {p_text(float(row['p']))}, {p_stars(float(row['p']))}."
        )
    change_key = stats_frame[
        (stats_frame["test"] == "Mann-Whitney WT vs HET change")
        & (stats_frame["metric"].isin(["call_rate_per_min", "vocal_output_s_per_min"]))
    ]
    lines.extend(["", "## Genotype Difference in Pre/Post Change"])
    for _, row in change_key.iterrows():
        lines.append(
            f"- {row['metric']}: WT mean change={row['mean_change_WT']:.4g}, HET mean change={row['mean_change_HET']:.4g}, {p_text(float(row['p']))}, {p_stars(float(row['p']))}."
        )
    _, pca_loadings, pca_variance = build_pca_tables(calls)
    pc1_top = pca_loadings.assign(magnitude=pca_loadings["PC1"].abs()).nlargest(3, "magnitude")
    pc2_top = pca_loadings.assign(magnitude=pca_loadings["PC2"].abs()).nlargest(3, "magnitude")
    pc3_top = pca_loadings.assign(magnitude=pca_loadings["PC3"].abs()).nlargest(3, "magnitude")
    lines.extend(
        [
            "",
            "## Animal-Level Call-Feature PCA",
            f"- PC1 explained {100.0 * float(pca_variance.iloc[0]['explained_variance_ratio']):.1f}%, PC2 explained {100.0 * float(pca_variance.iloc[1]['explained_variance_ratio']):.1f}%, and PC3 explained {100.0 * float(pca_variance.iloc[2]['explained_variance_ratio']):.1f}% of variance ({100.0 * float(pca_variance.iloc[2]['cumulative_explained_variance']):.1f}% cumulative).",
            "- Top absolute PC1 loadings: " + ", ".join(pc1_top["feature_label"].astype(str)) + ".",
            "- Top absolute PC2 loadings: " + ", ".join(pc2_top["feature_label"].astype(str)) + ".",
            "- Top absolute PC3 loadings: " + ", ".join(pc3_top["feature_label"].astype(str)) + ".",
            "- PCA used one row per animal and median values for 12 standardized, nonredundant call features.",
        ]
    )
    lines.extend(["", "## Figures"])
    if figures:
        for figure in figures:
            lines.append(f"- `{figure.name}`")
    else:
        lines.append("- Not regenerated in this run (`--no-figures`).")
    if len(missing):
        lines.extend(["", "## Missing Detector Outputs"])
        for _, row in missing.iterrows():
            lines.append(f"- {row['animal_id']}: `{row['stats_path']}`")
    lines.extend(
        [
            "",
            "## Caveat",
            "These statistics use upstream VocalPy detector/classifier outputs. They are much stronger than ad hoc thresholding, but the class repertoire still depends on the pretrained classifier and should be checked against representative spectrograms before biological over-interpretation.",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the LgDel WT versus HET analysis from detector CSVs."""

    global ANALYSIS_END_S
    args = parse_args(argv)
    if args.partner_window_seconds <= 0:
        raise ValueError("--partner-window-seconds must be positive")
    ANALYSIS_END_S = 300.0 + float(args.partner_window_seconds)
    PHASE_LIMITS["partner"] = (300.0, ANALYSIS_END_S)
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.cohort_manifest:
        labels_path = args.cohort_labels.resolve() if args.cohort_labels else None
        entries = load_cohort_manifest(args.cohort_manifest.resolve(), labels_path)
    else:
        entries = load_cohort(data_root, args.trial_plan.resolve(), args.completed_plan.resolve())
    manifest = build_manifest(entries)
    available_entries = [entry for entry in entries if entry.stats_path.exists()]
    calls = load_calls(available_entries)
    phase_summary = build_phase_summary(calls, available_entries)
    time_bins = build_time_bins(calls, available_entries, args.bin_seconds)
    main_stats = mann_whitney_table(phase_summary)
    paired_stats = paired_change_table(phase_summary)
    rep_stats = repertoire_stats(calls)
    stats_frame = pd.concat([main_stats, paired_stats, rep_stats], ignore_index=True)
    pca_scores, pca_loadings, pca_variance = build_pca_tables(calls)

    manifest.to_csv(output_dir / "manifest.csv", index=False)
    calls.to_csv(output_dir / "detected_calls_merged.csv", index=False)
    phase_summary.to_csv(output_dir / "animal_phase_summary.csv", index=False)
    time_bins.to_csv(output_dir / "time_bins_30s.csv", index=False)
    stats_frame.to_csv(output_dir / "statistics.csv", index=False)
    pca_scores.to_csv(output_dir / "pca_animal_scores.csv", index=False)
    pca_loadings.to_csv(output_dir / "pca_feature_loadings.csv", index=False)
    pca_variance.to_csv(output_dir / "pca_explained_variance.csv", index=False)
    figures = [] if args.no_figures else make_figure(calls, phase_summary, time_bins, stats_frame, output_dir)
    report = write_report(output_dir, manifest, calls, phase_summary, stats_frame, figures)
    print(f"loaded calls: {len(calls)}")
    print(f"output: {output_dir}")
    print(f"report: {report}")
    for figure in figures:
        print(f"figure: {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
