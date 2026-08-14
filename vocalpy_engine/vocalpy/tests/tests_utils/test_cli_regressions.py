# -*- coding: utf-8 -*-
"""Regression tests for CLI argument parsing and error handling."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import pytest

from vocalpy.cli import build_parser, main


def test_cli_uses_default_frequency_sentinels_without_parse_failure():
    args = build_parser().parse_args([])

    assert args.lower_frequency_cutoff == "default"
    assert args.higher_frequency_cutoff == "default"
    assert args.segmentation_threshold == "default"
    assert args.segmentation_model_path is None


def test_cli_parses_numeric_frequency_overrides_as_integers():
    args = build_parser().parse_args(["-lf", "45000", "-hf", "125000"])

    assert args.lower_frequency_cutoff == 45000
    assert args.higher_frequency_cutoff == 125000


def test_cli_exits_with_parser_error_for_invalid_input(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--path_to_audio", "/tmp/does-not-exist"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "error: audio path is not a file or directory" in captured.err


def test_cli_accepts_segmenter_without_explicit_model_path():
    args = build_parser().parse_args(["--segmenter"])

    assert args.segmenter is True
    assert args.segmentation_model_path is None
