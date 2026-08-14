# -*- coding: utf-8 -*-
"""Behavior tests for runtime argument validation helpers."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import logging
from argparse import Namespace

import pytest

from vocalpy.errors import ValidationError
from vocalpy.utils.misc import (
    create_logger,
    validate_animal,
    validate_arguments,
    validate_bin_size,
    validate_frequency_range,
    validate_thread_count,
)


@pytest.fixture
def cli_args():
    return Namespace(
        animal="mouse",
        path_to_audio="/tmp/audio.wav",
        bin_size=60,
        lower_frequency_cutoff="default",
        higher_frequency_cutoff="default",
        threads=-1,
        verbose=False,
        validation=False,
    )


def test_create_logger_writes_to_output_log(tmp_path):
    args = Namespace(verbose=False)

    logger = create_logger(args, tmp_path)
    logger.info("pipeline ready")

    output = (tmp_path / "output.log").read_text()
    assert logger is logging.getLogger()
    assert "logging to file:" in output
    assert "pipeline ready" in output


def test_create_logger_stays_quiet_without_verbose_stdout(tmp_path, capsys):
    create_logger(Namespace(verbose=False), tmp_path)

    captured = capsys.readouterr()

    assert captured.out == ""


def test_validate_arguments_applies_default_frequency_cutoffs(cli_args):
    validated = validate_arguments(cli_args)

    assert validated.lower_frequency_cutoff == 45000
    assert validated.higher_frequency_cutoff == 125000


def test_validate_arguments_supports_legacy_frequency_field_names():
    args = Namespace(
        animal="guineapig",
        bin_size=10,
        lower_frequency="default",
        higher_frequency="default",
        threads=1,
    )

    validated = validate_arguments(args)

    assert validated.lower_frequency == 0
    assert validated.higher_frequency == 22000


def test_validate_arguments_rejects_invalid_values(cli_args):
    cli_args.threads = 0

    with pytest.raises(ValidationError, match="number of threads"):
        validate_arguments(cli_args)


def test_validate_bin_size_requires_positive_integer():
    with pytest.raises(ValidationError, match="bin_size"):
        validate_bin_size(0)


def test_validate_frequency_range_supports_partial_defaults():
    low_frequency, high_frequency = validate_frequency_range("default", 90000, "mouse")

    assert (low_frequency, high_frequency) == (45000, 90000)


def test_validate_frequency_range_rejects_inverted_cutoffs():
    with pytest.raises(ValidationError, match="low frequency cutoff"):
        validate_frequency_range(50000, 20000, "mouse")


def test_validate_thread_count_rejects_invalid_values(monkeypatch):
    monkeypatch.setattr("vocalpy.utils.misc.cpu_count", lambda: 8)

    with pytest.raises(ValidationError, match="positive integer or -1"):
        validate_thread_count(-2)


def test_validate_thread_count_warns_when_request_exceeds_core_count(monkeypatch):
    monkeypatch.setattr("vocalpy.utils.misc.cpu_count", lambda: 4)

    with pytest.warns(UserWarning, match="equal or higher than number of available cores"):
        validate_thread_count(8)


def test_validate_animal_rejects_unknown_pipeline():
    with pytest.raises(ValidationError, match="unsupported animal pipeline"):
        validate_animal("mousee")
