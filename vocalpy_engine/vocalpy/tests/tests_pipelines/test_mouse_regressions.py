# -*- coding: utf-8 -*-
"""Regression tests for the mouse pipeline."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import importlib
import sys
import types

import numpy as np
import pytest


class StopAfterChunkParse(Exception):
    """Raised by the fake audio reader once chunk parsing succeeded."""


@pytest.fixture
def mouse_module(monkeypatch):
    fake_cv2 = types.ModuleType("cv2")
    fake_scipy = types.ModuleType("scipy")
    fake_scipy.ndimage = types.ModuleType("scipy.ndimage")
    fake_skimage = types.ModuleType("skimage")
    fake_skimage.exposure = types.ModuleType("skimage.exposure")
    fake_skimage.measure = types.ModuleType("skimage.measure")

    fake_io = types.ModuleType("vocalpy.utils.io")
    fake_io.read_audio = lambda *args, **kwargs: (_ for _ in ()).throw(StopAfterChunkParse())

    fake_classifier = types.ModuleType("vocalpy.nn.classifier")
    fake_classifier.VocalClassifier = object

    fake_datasets = types.ModuleType("vocalpy.nn.datasets")
    fake_datasets.create_array_from_list_of_vocals = lambda list_of_vocals: list_of_vocals

    fake_signal_processing = types.ModuleType("vocalpy.utils.signal_processing")
    fake_signal_processing.compute_spectrogram = lambda *args, **kwargs: None

    fake_image_processing = types.ModuleType("vocalpy.utils.image_processing")
    fake_image_processing.normalize = lambda data: data
    fake_image_processing.contrast_adjustment = lambda data, **kwargs: data
    fake_image_processing.bradley_roth = lambda data, **kwargs: data

    fake_vocal = types.ModuleType("vocalpy.modules.vocal")
    fake_vocal.Vocal = object

    fake_list_of_vocals = types.ModuleType("vocalpy.modules.list_of_vocals")
    fake_list_of_vocals.ListOfVocals = object

    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
    monkeypatch.setitem(sys.modules, "scipy.ndimage", fake_scipy.ndimage)
    monkeypatch.setitem(sys.modules, "skimage", fake_skimage)
    monkeypatch.setitem(sys.modules, "skimage.exposure", fake_skimage.exposure)
    monkeypatch.setitem(sys.modules, "skimage.measure", fake_skimage.measure)
    monkeypatch.setitem(sys.modules, "vocalpy.utils.io", fake_io)
    monkeypatch.setitem(sys.modules, "vocalpy.nn.classifier", fake_classifier)
    monkeypatch.setitem(sys.modules, "vocalpy.nn.datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "vocalpy.utils.signal_processing", fake_signal_processing)
    monkeypatch.setitem(sys.modules, "vocalpy.utils.image_processing", fake_image_processing)
    monkeypatch.setitem(sys.modules, "vocalpy.modules.vocal", fake_vocal)
    monkeypatch.setitem(sys.modules, "vocalpy.modules.list_of_vocals", fake_list_of_vocals)
    monkeypatch.delitem(sys.modules, "vocalpy.pipelines.mouse", raising=False)

    mouse_module = importlib.import_module("vocalpy.pipelines.mouse")
    animal_module = importlib.import_module("vocalpy.pipelines.animal")
    monkeypatch.setattr(animal_module, "read_audio", lambda *args, **kwargs: (_ for _ in ()).throw(StopAfterChunkParse()))

    return mouse_module


def test_identifier_does_not_require_removed_numpy_float_alias(mouse_module, monkeypatch):
    monkeypatch.setattr(mouse_module.np, "float", object(), raising=False)
    mouse = mouse_module.Mouse("mouse", {})
    chunk = np.array(["audio.wav", "out", "spec", "mask", "1000", "60", "1", "0", "100"])

    with pytest.raises(StopAfterChunkParse):
        mouse.identifier(chunk)
