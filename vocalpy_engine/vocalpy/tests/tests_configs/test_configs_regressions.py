# -*- coding: utf-8 -*-
"""Regression tests for configuration loading."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

from types import SimpleNamespace

import pytest

from vocalpy.configs.configs import read_default_parameters
from vocalpy.errors import ConfigurationError


def test_read_default_parameters_raises_explicit_error_for_unknown_animal():
    args = SimpleNamespace(animal="dragon")

    with pytest.raises(ConfigurationError, match="could not find animal pipeline"):
        read_default_parameters(args)
