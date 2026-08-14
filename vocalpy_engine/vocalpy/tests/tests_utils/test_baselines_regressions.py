# -*- coding: utf-8 -*-
"""Regression tests for example baseline comparison utilities."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

from pathlib import Path

import pandas as pd

from vocalpy.utils.baselines import (
    BaselineFixture,
    ComparisonResult,
    compare_fixture,
    format_result,
    format_totals,
    load_fixtures,
    summarize_results,
)


def test_load_fixtures_filters_manifest_by_species_and_name(tmp_path):
    manifest = tmp_path / "baselines.yml"
    manifest.write_text(
        """
fixtures:
  - name: mouse_1
    species: mouse
  - name: rat_1
    species: rat
"""
    )

    fixtures = load_fixtures(manifest, species=["mouse"], names=["mouse_1", "rat_1"])

    assert fixtures == [BaselineFixture(name="mouse_1", species="mouse", audio_filename="mouse_1.wav", outputs_dirname="mouse_1_outputs")]


def test_compare_fixture_ignores_class_columns_when_fixture_has_no_classifier(tmp_path):
    audio_dir = tmp_path / "examples"
    baseline_dir = audio_dir / "gp_1_outputs"
    current_dir = tmp_path / "work" / "gp_1" / "gp_1_outputs"
    baseline_dir.mkdir(parents=True)
    current_dir.mkdir(parents=True)

    dataframe = pd.DataFrame(
        {
            "start(s)": [1.0],
            "end(s)": [1.1],
            "duration(ms)": [100.0],
            "avg_intensity": [-20.0],
            "bg_intensity": [-30.0],
        }
    )
    dataframe.to_csv(baseline_dir / "gp_1_stats.csv", index=False)
    dataframe.to_csv(current_dir / "gp_1_stats.csv", index=False)

    result = compare_fixture(
        BaselineFixture(name="gp_1", species="guineapig", audio_filename="gp_1.wav", outputs_dirname="gp_1_outputs"),
        audio_dir=audio_dir,
        work_dir=tmp_path / "work",
        tolerance_ms=3.0,
    )

    assert result.top1_mismatches == 0
    assert result.top2_mismatches == 0


def test_format_result_renders_optional_detail_sections():
    result = ComparisonResult(
        fixture=BaselineFixture(name="mouse_1", species="mouse", audio_filename="mouse_1.wav", outputs_dirname="mouse_1_outputs"),
        baseline_count=3,
        current_count=4,
        matched_count=3,
        extra_rows=pd.DataFrame(
            {
                "start(s)": [1.0],
                "end(s)": [1.1],
                "duration(ms)": [100.0],
                "avg_intensity": [-20.0],
                "bg_intensity": [-30.0],
            }
        ),
        missing_rows=pd.DataFrame(),
        top1_mismatches=1,
        top2_mismatches=0,
        max_start_delta_ms=0.5,
        max_end_delta_ms=0.0,
        extra_image_paths=[Path("/tmp/extra.png")],
    )

    output = format_result(result)

    assert "== mouse_1 (mouse) ==" in output
    assert "extra rows:" in output
    assert "extra validation images:" in output
    assert "/tmp/extra.png" in output


def test_summarize_results_and_format_totals_aggregate_drift():
    results = [
        ComparisonResult(
            fixture=BaselineFixture(name="mouse_1", species="mouse", audio_filename="mouse_1.wav", outputs_dirname="mouse_1_outputs"),
            baseline_count=3,
            current_count=3,
            matched_count=3,
            extra_rows=pd.DataFrame(),
            missing_rows=pd.DataFrame(),
            top1_mismatches=0,
            top2_mismatches=0,
            max_start_delta_ms=0.0,
            max_end_delta_ms=0.0,
            extra_image_paths=[],
        ),
        ComparisonResult(
            fixture=BaselineFixture(name="mouse_2", species="mouse", audio_filename="mouse_2.wav", outputs_dirname="mouse_2_outputs"),
            baseline_count=54,
            current_count=55,
            matched_count=54,
            extra_rows=pd.DataFrame({"start(s)": [1.0]}),
            missing_rows=pd.DataFrame(),
            top1_mismatches=1,
            top2_mismatches=2,
            max_start_delta_ms=1.2,
            max_end_delta_ms=0.3,
            extra_image_paths=[],
        ),
    ]

    totals = summarize_results(results)
    output = format_totals(totals, Path("/tmp/work"))

    assert totals.extra_total == 1
    assert totals.top1_total == 1
    assert totals.top2_total == 2
    assert totals.drift_found is True
    assert "totals: extra=1 missing=0" in output
    assert "work_dir=/tmp/work" in output
