# -*- coding: utf-8 -*-
"""Regression tests for image-processing utilities."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import numpy as np

from vocalpy.utils import image_processing


def test_bradley_roth_does_not_require_removed_numpy_aliases(monkeypatch):
    monkeypatch.setattr(image_processing.np, "float", object(), raising=False)
    monkeypatch.setattr(image_processing.np, "int", object(), raising=False)
    image = np.array([[0.0, 0.2], [0.8, 1.0]])

    output = image_processing.bradley_roth(image, s=2, t=15)

    assert output.shape == image.shape
    assert output.dtype == np.uint8
