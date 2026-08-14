# -*- coding: utf-8 -*-
"""Utilities for validating shipped example baselines."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import argparse
import shutil
import subprocess
import sys

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import yaml


DEFAULT_MANIFEST = Path("examples/audios/baselines.yml")


@dataclass(frozen=True)
class BaselineFixture:
    name: str
    species: str
    audio_filename: str
    outputs_dirname: str


@dataclass
class ComparisonResult:
    fixture: BaselineFixture
    baseline_count: int
    current_count: int
    matched_count: int
    extra_rows: pd.DataFrame
    missing_rows: pd.DataFrame
    top1_mismatches: int
    top2_mismatches: int
    max_start_delta_ms: float
    max_end_delta_ms: float
    extra_image_paths: list[Path]


@dataclass(frozen=True)
class ComparisonTotals:
    extra_total: int
    missing_total: int
    top1_total: int
    top2_total: int
    max_start_delta_ms: float
    max_end_delta_ms: float
    drift_found: bool


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=Path("examples/audios"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--species", action="append", default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--tolerance-ms", type=float, default=3.0)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--fail-on-drift", action="store_true")
    parser.add_argument("--max-extra-total", type=int, default=None)
    parser.add_argument("--max-missing-total", type=int, default=0)
    parser.add_argument("--max-top1-mismatches-total", type=int, default=None)
    parser.add_argument("--max-top2-mismatches-total", type=int, default=None)
    parser.add_argument("--max-start-delta-ms", type=float, default=None)
    parser.add_argument("--max-end-delta-ms", type=float, default=None)
    parser.add_argument("files", nargs="*", default=None)
    return parser


def load_fixtures(manifest_path: str | Path, species=None, names=None) -> list[BaselineFixture]:
    manifest = yaml.safe_load(Path(manifest_path).read_text()) or {}
    manifest_fixtures = manifest.get("fixtures", [])
    fixtures = [
        BaselineFixture(
            name=fixture["name"],
            species=fixture["species"],
            audio_filename=fixture.get("audio_filename", f"{fixture['name']}.wav"),
            outputs_dirname=fixture.get("outputs_dirname", f"{fixture['name']}_outputs"),
        )
        for fixture in manifest_fixtures
    ]

    if species:
        allowed_species = set(species)
        fixtures = [fixture for fixture in fixtures if fixture.species in allowed_species]

    if names:
        allowed_names = set(names)
        fixtures = [fixture for fixture in fixtures if fixture.name in allowed_names]

    return fixtures


def run_pipeline_for_fixture(audio_dir: Path, work_dir: Path, fixture: BaselineFixture, validation: bool) -> None:
    source_audio = audio_dir / fixture.audio_filename
    run_dir = work_dir / fixture.name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_audio = run_dir / source_audio.name
    shutil.copy2(source_audio, run_audio)
    command = [sys.executable, "-m", "vocalpy.cli", "-a", fixture.species, "-p", str(run_audio)]
    if validation:
        command.append("-l")
    subprocess.run(command, check=True)


def align_rows(base: pd.DataFrame, current: pd.DataFrame, tolerance_seconds: float):
    matches = []
    used_current = set()
    for base_index, base_row in base.iterrows():
        matched_index = None
        for current_index, current_row in current.iterrows():
            if current_index in used_current:
                continue
            if (
                abs(base_row["start(s)"] - current_row["start(s)"]) <= tolerance_seconds
                and abs(base_row["end(s)"] - current_row["end(s)"]) <= tolerance_seconds
            ):
                matched_index = current_index
                break
        if matched_index is not None:
            used_current.add(matched_index)
            matches.append((base_index, matched_index))

    missing = base.drop(index=[base_index for base_index, _ in matches])
    extra = current.drop(index=list(used_current))
    return matches, missing, extra


def count_classification_mismatches(matched_baseline: pd.DataFrame, matched_current: pd.DataFrame, column_name: str) -> int:
    if column_name not in matched_baseline.columns or column_name not in matched_current.columns:
        return 0
    return int((matched_current[column_name] != matched_baseline[column_name]).sum())


def compare_fixture(fixture: BaselineFixture, audio_dir: Path, work_dir: Path, tolerance_ms: float) -> ComparisonResult:
    baseline = pd.read_csv(audio_dir / fixture.outputs_dirname / f"{fixture.name}_stats.csv")
    current = pd.read_csv(work_dir / fixture.name / fixture.outputs_dirname / f"{fixture.name}_stats.csv")
    matches, missing, extra = align_rows(baseline, current, tolerance_ms / 1000.0)

    if matches:
        matched_baseline = baseline.loc[[base_index for base_index, _ in matches]].reset_index(drop=True)
        matched_current = current.loc[[current_index for _, current_index in matches]].reset_index(drop=True)
        max_start_delta_ms = float((matched_current["start(s)"] - matched_baseline["start(s)"]).abs().max() * 1000)
        max_end_delta_ms = float((matched_current["end(s)"] - matched_baseline["end(s)"]).abs().max() * 1000)
        top1_mismatches = count_classification_mismatches(matched_baseline, matched_current, "class_top1")
        top2_mismatches = count_classification_mismatches(matched_baseline, matched_current, "class_top2")
    else:
        max_start_delta_ms = 0.0
        max_end_delta_ms = 0.0
        top1_mismatches = 0
        top2_mismatches = 0

    validation_dir = work_dir / fixture.name / fixture.outputs_dirname / "spectrogram_validation"
    extra_image_paths = []
    if validation_dir.exists():
        for current_index in extra.index.tolist():
            extra_image_paths.extend(sorted(validation_dir.glob(f"{current_index + 1}_*.png")))

    return ComparisonResult(
        fixture=fixture,
        baseline_count=len(baseline),
        current_count=len(current),
        matched_count=len(matches),
        extra_rows=extra,
        missing_rows=missing,
        top1_mismatches=top1_mismatches,
        top2_mismatches=top2_mismatches,
        max_start_delta_ms=max_start_delta_ms,
        max_end_delta_ms=max_end_delta_ms,
        extra_image_paths=extra_image_paths,
    )


def get_detail_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.columns
        if column in {"start(s)", "end(s)", "duration(ms)", "avg_intensity", "bg_intensity", "class_top1", "class_top2"}
    ]


def format_result(result: ComparisonResult) -> str:
    lines = [
        f"== {result.fixture.name} ({result.fixture.species}) ==",
        (
            f"baseline={result.baseline_count} current={result.current_count} "
            f"matched={result.matched_count} extra={len(result.extra_rows)} missing={len(result.missing_rows)}"
        ),
        (
            f"max_start_delta_ms={result.max_start_delta_ms:.3f} "
            f"max_end_delta_ms={result.max_end_delta_ms:.3f} "
            f"class_top1_mismatches={result.top1_mismatches} "
            f"class_top2_mismatches={result.top2_mismatches}"
        ),
    ]
    if not result.extra_rows.empty:
        columns = get_detail_columns(result.extra_rows)
        lines.append("extra rows:")
        lines.append(result.extra_rows[columns].to_string(index=False))
        if result.extra_image_paths:
            lines.append("extra validation images:")
            for path in result.extra_image_paths:
                lines.append(str(path))
    if not result.missing_rows.empty:
        columns = get_detail_columns(result.missing_rows)
        lines.append("missing rows:")
        lines.append(result.missing_rows[columns].to_string(index=False))
    return "\n".join(lines)


def summarize_results(results: list[ComparisonResult]) -> ComparisonTotals:
    extra_total = 0
    missing_total = 0
    top1_total = 0
    top2_total = 0
    max_start_delta_ms = 0.0
    max_end_delta_ms = 0.0
    drift_found = False

    for result in results:
        extra_total += result.extra_rows.shape[0]
        missing_total += result.missing_rows.shape[0]
        top1_total += result.top1_mismatches
        top2_total += result.top2_mismatches
        max_start_delta_ms = max(max_start_delta_ms, result.max_start_delta_ms)
        max_end_delta_ms = max(max_end_delta_ms, result.max_end_delta_ms)
        if result.extra_rows.shape[0] or result.missing_rows.shape[0] or result.top1_mismatches or result.top2_mismatches:
            drift_found = True

    return ComparisonTotals(
        extra_total=extra_total,
        missing_total=missing_total,
        top1_total=top1_total,
        top2_total=top2_total,
        max_start_delta_ms=max_start_delta_ms,
        max_end_delta_ms=max_end_delta_ms,
        drift_found=drift_found,
    )


def format_totals(totals: ComparisonTotals, work_dir: Path) -> str:
    return "\n".join(
        [
            (
                "totals: "
                f"extra={totals.extra_total} missing={totals.missing_total} "
                f"top1_mismatches={totals.top1_total} top2_mismatches={totals.top2_total} "
                f"max_start_delta_ms={totals.max_start_delta_ms:.3f} "
                f"max_end_delta_ms={totals.max_end_delta_ms:.3f}"
            ),
            f"work_dir={work_dir}",
        ]
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    fixtures = load_fixtures(args.manifest, species=args.species, names=args.files)
    if not fixtures:
        print("no matching baseline fixtures found")
        return 1

    temp_dir = None
    work_dir = args.work_dir
    if work_dir is None:
        temp_dir = TemporaryDirectory(prefix="vocalpy-example-compare-")
        work_dir = Path(temp_dir.name)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        for fixture in fixtures:
            run_pipeline_for_fixture(args.audio_dir, work_dir, fixture, args.validation)

        results = [compare_fixture(fixture, args.audio_dir, work_dir, args.tolerance_ms) for fixture in fixtures]
        for result in results:
            print(format_result(result))
            print()
        totals = summarize_results(results)
        print(format_totals(totals, work_dir))
        if args.max_extra_total is not None and totals.extra_total > args.max_extra_total:
            return 1
        if args.max_missing_total is not None and totals.missing_total > args.max_missing_total:
            return 1
        if args.max_top1_mismatches_total is not None and totals.top1_total > args.max_top1_mismatches_total:
            return 1
        if args.max_top2_mismatches_total is not None and totals.top2_total > args.max_top2_mismatches_total:
            return 1
        if args.max_start_delta_ms is not None and totals.max_start_delta_ms > args.max_start_delta_ms:
            return 1
        if args.max_end_delta_ms is not None and totals.max_end_delta_ms > args.max_end_delta_ms:
            return 1
        return 1 if (args.fail_on_drift and totals.drift_found) else 0
    finally:
        if temp_dir is not None and not args.keep_work_dir:
            temp_dir.cleanup()
