"""Analyze LgDel close-loop USV detections as WT versus HET, ignoring virus.

Inputs are the upstream MiMic/VocalPy ``*_stats.csv`` files created next to each
WAV. The requested design is encoded directly:

* alone: 0 to 300 s
* partner: 300 to 900 s
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy import stats
from scipy.stats import chi2_contingency

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vpgui.engine import CLASS_COLORS

PHASE_ORDER = ["alone", "partner"]
PHASE_LABEL = {"alone": "alone", "partner": "+ partner"}
PHASE_LIMITS = {"alone": (0.0, 300.0), "partner": (300.0, 900.0)}
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
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "lgdel_usv_analysis")
    parser.add_argument("--bin-seconds", type=float, default=30.0)
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
    if time_s < 900.0:
        return "partner"
    return "outside"


def load_cohort(data_root: pathlib.Path, trial_plan: pathlib.Path, completed_plan: pathlib.Path) -> list[CohortEntry]:
    """Load trial annotations, preferring completed genotype and virus labels."""

    plan = pd.read_csv(trial_plan)
    completed = pd.read_excel(completed_plan)
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

    edges = np.arange(0.0, 900.0 + bin_seconds, bin_seconds)
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


def apply_style() -> None:
    """Apply a clean publication-style Matplotlib theme."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "font.family": ["DejaVu Sans"],
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
    ax.set_title("Syllable repertoire composition")
    for xpos, (genotype, phase) in zip(x, cols):
        n_calls = len(calls[(calls["genotype"] == genotype) & (calls["phase"] == phase)]) if not calls.empty else 0
        ax.text(xpos, 1.02, f"n={n_calls}", ha="center", va="bottom", fontsize=8)
    for phase, pair in {"alone": (0, 1), "partner": (2, 3)}.items():
        annotate_bracket(ax, pair[0], pair[1], 1.08, lookup_p(stats_frame, phase, "class_top1"))
    ax.legend(title="class", fontsize=7, title_fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left")


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
    ax.set_xticklabels(totals["animal_id"], rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("total calls")
    ax.set_title("Total vocal output per mouse")


def plot_call_rate(ax: Axes, phase_summary: pd.DataFrame, stats_frame: pd.DataFrame) -> None:
    """Plot animal-level call rate by genotype and phase."""

    x = np.arange(len(PHASE_ORDER))
    width = 0.34
    for offset, genotype in [(-width / 2, "WT"), (width / 2, "HET")]:
        means = []
        errors = []
        for phase in PHASE_ORDER:
            values = phase_summary.loc[
                (phase_summary["phase"] == phase) & (phase_summary["genotype"] == genotype),
                "call_rate_per_min",
            ].to_numpy(dtype=float)
            means.append(float(np.mean(values)) if len(values) else np.nan)
            errors.append(sem(values))
        label = f"{GENOTYPE_LABEL[genotype]} (n={phase_summary[phase_summary['genotype'] == genotype]['animal_id'].nunique()})"
        ax.bar(x + offset, means, width, yerr=errors, capsize=3, color=GENOTYPE_COLOR[genotype], edgecolor="white", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABEL[p] for p in PHASE_ORDER])
    ax.set_ylabel("calls / min")
    ax.set_title("Call rate per genotype x phase")
    ymax = ax.get_ylim()[1]
    for index, phase in enumerate(PHASE_ORDER):
        annotate_bracket(ax, index - width / 2, index + width / 2, ymax * (0.82 + index * 0.08), lookup_p(stats_frame, phase, "call_rate_per_min"))
    ax.legend(loc="upper left", fontsize=8)


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
    ax.set_xlim(0, 900)
    ax.set_xlabel("time in session (s)")
    ax.set_ylabel("calls / 30 s bin")
    ax.set_title("Call timeline: partner introduction at 5 min")
    ax.legend(loc="upper right")


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
    ax.set_ylabel("call duration (s / min)")
    ax.set_title("Vocal output by social phase")
    ymax = ax.get_ylim()[1]
    for index, phase in enumerate(PHASE_ORDER):
        annotate_bracket(ax, index - 0.10, index + 0.10, ymax * (0.78 + index * 0.10), lookup_p(stats_frame, phase, "vocal_output_s_per_min"))
    ax.legend(loc="upper left")


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
    ax.set_xlabel("mean duration (ms)")
    ax.set_ylabel("mean avg frequency (kHz)")
    ax.set_title("Call acoustic features")
    ax.legend(loc="best", fontsize=8)


def make_figure(calls: pd.DataFrame, phase_summary: pd.DataFrame, time_bins: pd.DataFrame, stats_frame: pd.DataFrame, output_dir: pathlib.Path) -> list[pathlib.Path]:
    """Create the requested multi-panel USV summary figure."""

    apply_style()
    fig = plt.figure(figsize=(11.5, 14.0), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=[1.15, 1.0, 1.05, 1.0])
    plot_repertoire(fig.add_subplot(grid[0, 0]), calls, stats_frame)
    plot_total_calls(fig.add_subplot(grid[0, 1]), phase_summary)
    plot_call_rate(fig.add_subplot(grid[1, 0]), phase_summary, stats_frame)
    plot_acoustics(fig.add_subplot(grid[1, 1]), calls)
    plot_timeline(fig.add_subplot(grid[2, :]), time_bins)
    plot_vocal_output(fig.add_subplot(grid[3, :]), phase_summary, stats_frame)
    fig.suptitle("LgDel USV analysis: WT vs HET grouped across virus", fontsize=15, fontweight="bold")
    paths = [
        output_dir / "lgdel_usv_wt_vs_het_summary.png",
        output_dir / "lgdel_usv_wt_vs_het_summary.svg",
        output_dir / "lgdel_usv_wt_vs_het_summary.pdf",
    ]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
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
        "- Partner phase: 300 to 900 s.",
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
    lines.extend(["", "## Figures"])
    for figure in figures:
        lines.append(f"- `{figure.name}`")
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

    args = parse_args(argv)
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = load_cohort(data_root, args.trial_plan.resolve(), args.completed_plan.resolve())
    manifest = build_manifest(entries)
    available_entries = [entry for entry in entries if entry.stats_path.exists()]
    calls = load_calls(available_entries)
    phase_summary = build_phase_summary(calls, available_entries)
    time_bins = build_time_bins(calls, available_entries, args.bin_seconds)
    main_stats = mann_whitney_table(phase_summary)
    rep_stats = repertoire_stats(calls)
    stats_frame = pd.concat([main_stats, rep_stats], ignore_index=True)

    manifest.to_csv(output_dir / "manifest.csv", index=False)
    calls.to_csv(output_dir / "detected_calls_merged.csv", index=False)
    phase_summary.to_csv(output_dir / "animal_phase_summary.csv", index=False)
    time_bins.to_csv(output_dir / "time_bins_30s.csv", index=False)
    stats_frame.to_csv(output_dir / "statistics.csv", index=False)
    figures = make_figure(calls, phase_summary, time_bins, stats_frame, output_dir)
    report = write_report(output_dir, manifest, calls, phase_summary, stats_frame, figures)
    print(f"loaded calls: {len(calls)}")
    print(f"output: {output_dir}")
    print(f"report: {report}")
    for figure in figures:
        print(f"figure: {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
