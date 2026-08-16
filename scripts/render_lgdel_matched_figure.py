"""Render the matched-window LgDel USV summary without Matplotlib.

The renderer consumes the CSV outputs from ``lgdel_usv_analysis.py`` and writes
an SVG with the same six-panel structure as the original summary figure. A
local Chromium browser can rasterize the SVG to PNG and print it to PDF.
"""

from __future__ import annotations

import argparse
import html
import math
import pathlib
import subprocess

import numpy as np
import pandas as pd


WIDTH, HEIGHT = 1600, 1800
WT, HET = "#1f77b4", "#ff7f0e"
GRID, AXIS, TEXT, MUTED = "#e4e4e4", "#333333", "#111111", "#666666"
CLASS_COLORS = {
    "chevron": "#38bdf8", "complex": "#a78bfa", "down_fm": "#f472b6",
    "flat": "#34d399", "mult_steps": "#fbbf24", "rev_chevron": "#22d3ee",
    "short": "#fb7185", "step_down": "#818cf8", "step_up": "#2dd4bf",
    "two_steps": "#f59e0b", "up_fm": "#4ade80",
}
CLASSES = list(CLASS_COLORS)


def esc(value: object) -> str:
    return html.escape(str(value))


class Svg:
    def __init__(self) -> None:
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#111} .title{font-weight:700} .tick{font-size:13px} .label{font-size:16px} .panel{font-size:19px;font-weight:700}</style>',
        ]

    def add(self, value: str) -> None:
        self.parts.append(value)

    def text(self, x: float, y: float, value: object, size: int = 14, anchor: str = "middle", weight: int = 400, rotate: float | None = None, color: str = TEXT) -> None:
        transform = f' transform="rotate({rotate:g} {x:g} {y:g})"' if rotate is not None else ""
        self.add(f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}"{transform}>{esc(value)}</text>')

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = AXIS, width: float = 1.5, dash: str | None = None, opacity: float = 1.0) -> None:
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{color}" stroke-width="{width:g}" opacity="{opacity:g}"{dashed}/>')

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", opacity: float = 1.0) -> None:
        self.add(f'<rect x="{x:g}" y="{y:g}" width="{max(w, 0):g}" height="{max(h, 0):g}" fill="{fill}" stroke="{stroke}" opacity="{opacity:g}"/>')

    def circle(self, x: float, y: float, r: float, fill: str, opacity: float = 1.0) -> None:
        self.add(f'<circle cx="{x:g}" cy="{y:g}" r="{r:g}" fill="{fill}" stroke="white" stroke-width="1" opacity="{opacity:g}"/>')

    def polyline(self, points: list[tuple[float, float]], color: str, width: float = 2.5, opacity: float = 1.0, fill: str = "none") -> None:
        coords = " ".join(f"{x:g},{y:g}" for x, y in points)
        self.add(f'<polyline points="{coords}" fill="{fill}" stroke="{color}" stroke-width="{width:g}" stroke-linejoin="round" stroke-linecap="round" opacity="{opacity:g}"/>')

    def polygon(self, points: list[tuple[float, float]], fill: str, opacity: float = 0.2) -> None:
        coords = " ".join(f"{x:g},{y:g}" for x, y in points)
        self.add(f'<polygon points="{coords}" fill="{fill}" opacity="{opacity:g}"/>')

    def save(self, path: pathlib.Path) -> None:
        path.write_text("\n".join(self.parts + ["</svg>"]), encoding="utf-8")


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def nice_ticks(maximum: float, count: int = 5) -> list[float]:
    maximum = max(float(maximum), 1e-9)
    raw = maximum / count
    power = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 2.5, 5, 10), key=lambda x: abs(x * power - raw)) * power
    top = math.ceil(maximum / step) * step
    return [i * step for i in range(int(round(top / step)) + 1)]


def axes(svg: Svg, box: tuple[float, float, float, float], xmax: float, ymax: float, xticks: list[float], yticks: list[float], xlabel: str = "", ylabel: str = "", ytick_offset: float = 0.0):
    x, y, w, h = box
    left, top, pw, ph = x + 65, y + 45, w - 85, h - 95
    sx = lambda value: left + value / xmax * pw
    sy = lambda value: top + ph - value / ymax * ph
    for tick in yticks:
        yy = sy(tick)
        svg.line(left, yy, left + pw, yy, GRID, 1)
        svg.text(left - 10, yy + 5, f"{tick + ytick_offset:g}", 13, "end", color=MUTED)
    for tick in xticks:
        xx = sx(tick)
        svg.line(xx, top, xx, top + ph, GRID, 1)
        svg.text(xx, top + ph + 24, f"{tick:g}", 13, color=MUTED)
    svg.line(left, top, left, top + ph, AXIS, 1.5)
    svg.line(left, top + ph, left + pw, top + ph, AXIS, 1.5)
    if xlabel:
        svg.text(left + pw / 2, y + h - 10, xlabel, 16)
    if ylabel:
        svg.text(x + 16, top + ph / 2, ylabel, 16, rotate=-90)
    return sx, sy, (left, top, pw, ph)


def p_label(stats: pd.DataFrame, test: str, phase: str, metric: str) -> str:
    row = stats[(stats.test == test) & (stats.phase == phase) & (stats.metric == metric)]
    if row.empty:
        return "p=n/a"
    p = float(row.iloc[0].p)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    value = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    return f"{stars}  {value}"


def bracket(svg: Svg, x1: float, x2: float, y: float, label: str) -> None:
    svg.line(x1, y + 8, x1, y, AXIS, 1.5)
    svg.line(x1, y, x2, y, AXIS, 1.5)
    svg.line(x2, y, x2, y + 8, AXIS, 1.5)
    svg.text((x1 + x2) / 2, y - 7, label, 13)


def panel_repertoire(svg: Svg, calls: pd.DataFrame, stats: pd.DataFrame, box: tuple[float, float, float, float]) -> None:
    x, y, w, h = box
    svg.text(x + w / 2, y + 22, "Syllable repertoire composition", 19, weight=700)
    plot_box = (x, y, w - 125, h)
    sx, sy, plot = axes(svg, plot_box, 4, 1.12, [], [0, .2, .4, .6, .8, 1.0], ylabel="within-column proportion")
    left, top, pw, ph = plot
    combos = [("WT", "alone"), ("HET", "alone"), ("WT", "partner"), ("HET", "partner")]
    centers = [left + pw * (i + .5) / 4 for i in range(4)]
    bw = pw / 4 * .72
    for center, (genotype, phase) in zip(centers, combos):
        sub = calls[(calls.genotype == genotype) & (calls.phase == phase)]
        bottom = 0.0
        for cls in CLASSES:
            value = float((sub.class_top1 == cls).sum()) / max(len(sub), 1)
            svg.rect(center - bw / 2, sy(bottom + value), bw, sy(bottom) - sy(bottom + value), CLASS_COLORS[cls], "white")
            bottom += value
        svg.text(center, top + ph + 23, "WT" if genotype == "WT" else "Het", 14)
        svg.text(center, top + ph + 41, "alone" if phase == "alone" else "+ partner", 14)
        svg.text(center, sy(1.02), f"n={len(sub)}", 12)
    bracket(svg, centers[0], centers[1], sy(1.07), p_label(stats, "chi-square class distribution WT vs HET", "alone", "class_top1"))
    bracket(svg, centers[2], centers[3], sy(1.07), p_label(stats, "chi-square class distribution WT vs HET", "partner", "class_top1"))
    lx = left + pw + 12
    for i, cls in enumerate(CLASSES):
        yy = top + 14 + i * 20
        svg.rect(lx, yy - 10, 22, 10, CLASS_COLORS[cls])
        svg.text(lx + 29, yy, cls, 11, "start")


def panel_totals(svg: Svg, summary: pd.DataFrame, box: tuple[float, float, float, float]) -> None:
    x, y, w, h = box
    svg.text(x + w / 2, y + 22, "Total vocal output per mouse", 19, weight=700)
    totals = summary.groupby(["animal_id", "genotype"], as_index=False).call_count.sum().sort_values(["genotype", "call_count"], ascending=[True, False])
    ymax = nice_ticks(totals.call_count.max())[-1]
    sx, sy, plot = axes(svg, box, len(totals), ymax, [], nice_ticks(ymax), ylabel="total calls")
    left, top, pw, ph = plot
    bw = pw / len(totals) * .72
    for i, row in totals.reset_index(drop=True).iterrows():
        cx = left + pw * (i + .5) / len(totals)
        yy = sy(row.call_count)
        color = WT if row.genotype == "WT" else HET
        svg.rect(cx - bw / 2, yy, bw, top + ph - yy, color)
        svg.text(cx, yy - 6, int(row.call_count), 11)
        svg.text(cx, top + ph + 30, row.animal_id, 11, rotate=-58)


def panel_rates(svg: Svg, summary: pd.DataFrame, stats: pd.DataFrame, box: tuple[float, float, float, float]) -> None:
    x, y, w, h = box
    svg.text(x + w / 2, y + 22, "Call rate per genotype x phase", 19, weight=700)
    ymax_raw = summary.groupby(["genotype", "phase"]).call_rate_per_min.agg(["mean", sem]).eval("mean + sem").max() * 1.24
    yticks = nice_ticks(ymax_raw)
    ymax = yticks[-1]
    sx, sy, plot = axes(svg, box, 2, ymax, [], yticks, ylabel="calls / min")
    left, top, pw, ph = plot
    centers = [left + pw * .25, left + pw * .75]
    barw = pw * .18
    for pi, phase in enumerate(["alone", "partner"]):
        for gi, (genotype, color) in enumerate([("WT", WT), ("HET", HET)]):
            vals = summary[(summary.phase == phase) & (summary.genotype == genotype)].call_rate_per_min.to_numpy(float)
            mean, err = vals.mean(), sem(vals)
            cx = centers[pi] + (-barw / 2 if gi == 0 else barw / 2)
            svg.rect(cx - barw / 2, sy(mean), barw, top + ph - sy(mean), color)
            svg.line(cx, sy(mean - err), cx, sy(mean + err), AXIS, 2)
            svg.line(cx - 5, sy(mean - err), cx + 5, sy(mean - err), AXIS, 2)
            svg.line(cx - 5, sy(mean + err), cx + 5, sy(mean + err), AXIS, 2)
        svg.text(centers[pi], top + ph + 28, "alone" if phase == "alone" else "+ partner", 15)
        bracket(svg, centers[pi] - barw, centers[pi] + barw, sy(ymax * (.91 if pi == 0 else .96)), p_label(stats, "Mann-Whitney WT vs HET", phase, "call_rate_per_min"))
    svg.line(left + 12, top + 13, left + 37, top + 13, WT, 8)
    svg.text(left + 45, top + 18, "WT (n=12)", 12, "start")
    svg.line(left + 12, top + 34, left + 37, top + 34, HET, 8)
    svg.text(left + 45, top + 39, "Het (n=12)", 12, "start")


def panel_acoustics(svg: Svg, calls: pd.DataFrame, box: tuple[float, float, float, float]) -> None:
    x, y, w, h = box
    svg.text(x + w / 2, y + 22, "Call acoustic features", 19, weight=700)
    data = calls.groupby(["animal_id", "genotype"], as_index=False).agg(duration=("duration(ms)", "mean"), frequency=("avg_freq_khz", "mean"))
    xmin, xmax = 0.0, math.ceil(data.duration.max() / 5) * 5
    ymin = math.floor(data.frequency.min() / 5) * 5
    ymax = math.ceil(data.frequency.max() / 5) * 5
    shifted = data.copy()
    shifted["fy"] = shifted.frequency - ymin
    sx, sy, plot = axes(svg, box, xmax, ymax - ymin, list(np.arange(0, xmax + 1, 5)), list(np.arange(0, ymax - ymin + 1, 5)), xlabel="mean duration (ms)", ylabel="mean avg frequency (kHz)", ytick_offset=ymin)
    left, top, pw, ph = plot
    for _, row in shifted.iterrows():
        svg.circle(sx(row.duration), sy(row.fy), 6, WT if row.genotype == "WT" else HET, .9)
    svg.circle(left + pw - 80, top + 15, 5, WT)
    svg.text(left + pw - 67, top + 20, "WT", 12, "start")
    svg.circle(left + pw - 80, top + 36, 5, HET)
    svg.text(left + pw - 67, top + 41, "Het", 12, "start")


def panel_timeline(svg: Svg, bins: pd.DataFrame, box: tuple[float, float, float, float]) -> None:
    x, y, w, h = box
    svg.text(x + w / 2, y + 22, "Call timeline: partner introduction at 5 min", 19, weight=700)
    grouped = bins.groupby(["genotype", "bin_mid_s"]).call_count.agg(["mean", sem]).reset_index()
    ymax = nice_ticks((grouped["mean"] + grouped["sem"]).max())[-1]
    sx, sy, plot = axes(svg, box, 600, ymax, list(range(0, 601, 100)), nice_ticks(ymax), xlabel="time in session (s)", ylabel="calls / 30 s bin")
    left, top, pw, ph = plot
    for genotype, color in [("WT", WT), ("HET", HET)]:
        sub = grouped[grouped.genotype == genotype].sort_values("bin_mid_s")
        upper = [(sx(r.bin_mid_s), sy(r["mean"] + r["sem"])) for _, r in sub.iterrows()]
        lower = [(sx(r.bin_mid_s), sy(max(r["mean"] - r["sem"], 0))) for _, r in sub.iloc[::-1].iterrows()]
        svg.polygon(upper + lower, color, .18)
        svg.polyline([(sx(r.bin_mid_s), sy(r["mean"])) for _, r in sub.iterrows()], color, 2.6)
    intro = sx(300)
    svg.line(intro, top, intro, top + ph, MUTED, 1.5, "5 4")
    svg.text(intro + 10, top + 20, "+ partner", 13, "start", color=MUTED)
    svg.line(left + pw - 80, top + 15, left + pw - 52, top + 15, WT, 2.6)
    svg.text(left + pw - 42, top + 20, "WT", 13, "start")
    svg.line(left + pw - 80, top + 36, left + pw - 52, top + 36, HET, 2.6)
    svg.text(left + pw - 42, top + 41, "Het", 13, "start")


def panel_output(svg: Svg, summary: pd.DataFrame, stats: pd.DataFrame, box: tuple[float, float, float, float]) -> None:
    x, y, w, h = box
    svg.text(x + w / 2, y + 22, "Vocal output by social phase", 19, weight=700)
    ymax_raw = summary.vocal_output_s_per_min.max() * 1.12
    ymax = nice_ticks(ymax_raw)[-1]
    sx, sy, plot = axes(svg, box, 1.25, ymax, [], nice_ticks(ymax), ylabel="call duration (s / min)")
    left, top, pw, ph = plot
    xpos = [sx(.2), sx(1.05)]
    for genotype, color in [("WT", WT), ("HET", HET)]:
        sub = summary[summary.genotype == genotype]
        for _, animal in sub.groupby("animal_id"):
            values = [float(animal[animal.phase == phase].vocal_output_s_per_min.iloc[0]) for phase in ["alone", "partner"]]
            svg.polyline([(xpos[0], sy(values[0])), (xpos[1], sy(values[1]))], color, 1, .18)
        means, errors = [], []
        for phase in ["alone", "partner"]:
            values = sub[sub.phase == phase].vocal_output_s_per_min.to_numpy(float)
            means.append(values.mean()); errors.append(sem(values))
        svg.polyline([(xpos[0], sy(means[0])), (xpos[1], sy(means[1]))], color, 3)
        for xx, mean, error in zip(xpos, means, errors):
            svg.circle(xx, sy(mean), 7, color)
            svg.line(xx, sy(mean - error), xx, sy(mean + error), color, 2)
            svg.line(xx - 7, sy(mean - error), xx + 7, sy(mean - error), color, 2)
            svg.line(xx - 7, sy(mean + error), xx + 7, sy(mean + error), color, 2)
    svg.text(xpos[0], top + ph + 28, "alone", 15)
    svg.text(xpos[1], top + ph + 28, "+ partner", 15)
    bracket(svg, xpos[0] - 100, xpos[0] + 100, sy(ymax * .92), p_label(stats, "Mann-Whitney WT vs HET", "alone", "vocal_output_s_per_min"))
    bracket(svg, xpos[1] - 100, xpos[1] + 100, sy(ymax * .92), p_label(stats, "Mann-Whitney WT vs HET", "partner", "vocal_output_s_per_min"))
    svg.line(left + 15, top + 15, left + 43, top + 15, WT, 3)
    svg.text(left + 52, top + 20, "WT", 13, "start")
    svg.line(left + 15, top + 37, left + 43, top + 37, HET, 3)
    svg.text(left + 52, top + 42, "Het", 13, "start")


def render(input_dir: pathlib.Path, output_svg: pathlib.Path) -> None:
    calls = pd.read_csv(input_dir / "detected_calls_merged.csv")
    summary = pd.read_csv(input_dir / "animal_phase_summary.csv")
    bins = pd.read_csv(input_dir / "time_bins_30s.csv")
    stats = pd.read_csv(input_dir / "statistics.csv")
    svg = Svg()
    svg.text(WIDTH / 2, 36, "LgDel USV analysis: WT vs HET, matched 5 min windows", 27, weight=700)
    panel_repertoire(svg, calls, stats, (45, 55, 760, 410))
    panel_totals(svg, summary, (815, 55, 740, 410))
    panel_rates(svg, summary, stats, (45, 485, 760, 390))
    panel_acoustics(svg, calls, (815, 485, 740, 390))
    panel_timeline(svg, bins, (45, 900, 1510, 390))
    panel_output(svg, summary, stats, (45, 1320, 1510, 430))
    svg.save(output_svg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-svg", type=pathlib.Path, required=True)
    parser.add_argument("--browser", type=pathlib.Path)
    args = parser.parse_args()
    output = args.output_svg.resolve()
    render(args.input_dir.resolve(), output)
    if args.browser:
        uri = output.as_uri()
        png = output.with_suffix(".png")
        pdf = output.with_suffix(".pdf")
        subprocess.run([str(args.browser), "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", f"--window-size={WIDTH},{HEIGHT}", f"--screenshot={png}", uri], check=True)
        subprocess.run([str(args.browser), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", uri], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
