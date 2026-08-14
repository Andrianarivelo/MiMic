# -*- coding: utf-8 -*-
"""Regression tests for list-of-vocals behavior."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def list_of_vocals_module(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.tensor = lambda values: values
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delitem(sys.modules, "vocalpy.modules.list_of_vocals", raising=False)
    return importlib.import_module("vocalpy.modules.list_of_vocals")


def make_list_of_vocals(list_of_vocals_module, vocals):
    list_of_vocals = list_of_vocals_module.ListOfVocals()
    list_of_vocals.vocals_in_recording = list(vocals)
    list_of_vocals.number_of_vocals = len(vocals)
    return list_of_vocals


def test_update_centroids_does_not_require_removed_numpy_int_alias(list_of_vocals_module, monkeypatch):
    monkeypatch.setattr(list_of_vocals_module.np, "int", object(), raising=False)
    vocal = SimpleNamespace(
        start_coord=2,
        end_coord=6,
        min_freq_coord=4,
        max_freq_coord=8,
        centroid=None,
    )
    list_of_vocals = make_list_of_vocals(list_of_vocals_module, [vocal])

    list_of_vocals.update_centroids()

    assert vocal.centroid.tolist() == [6, 4]


def test_update_coords_does_not_require_removed_numpy_int_alias(list_of_vocals_module, monkeypatch):
    monkeypatch.setattr(list_of_vocals_module.np, "int", object(), raising=False)
    vocal = SimpleNamespace(coords=np.array([[0, 10], [0, 12], [0, 14]], dtype=int))
    list_of_vocals = make_list_of_vocals(list_of_vocals_module, [vocal])

    list_of_vocals.update_coords(spec_range=4)

    assert vocal.coords[:, 1].tolist() == [2, 4, 6]


def test_add_spectrograms_to_vocals_preserves_requested_width_at_left_edge(list_of_vocals_module):
    vocal = SimpleNamespace(centroid=[1, 1], spectrogram=None, mask=None)
    list_of_vocals = make_list_of_vocals(list_of_vocals_module, [vocal])
    full_spectrogram = np.arange(40, dtype=np.uint8).reshape(4, 10)
    full_mask = np.zeros_like(full_spectrogram)

    list_of_vocals.add_spectrograms_to_vocals(full_spectrogram, full_mask, spec_range=4)

    assert vocal.spectrogram.shape == (4, 8)
    assert np.array_equal(vocal.spectrogram[:, :3], np.zeros((4, 3), dtype=np.uint8))
    assert np.array_equal(vocal.spectrogram[:, 3:], full_spectrogram[:, :5])
    assert np.array_equal(vocal.mask[:, :3], np.zeros((4, 3), dtype=np.uint8))
    assert np.array_equal(vocal.mask[:, 3:], full_mask[:, :5])
    assert vocal.centroid == [1, 4]


def test_list_of_vocals_defaults_to_empty_array_state(list_of_vocals_module):
    list_of_vocals = list_of_vocals_module.ListOfVocals()

    assert list_of_vocals.number_of_vocals == 0
    assert list_of_vocals.vocals_in_recording.tolist() == []


def test_add_segmentation_masks_to_vocals_assigns_masks(list_of_vocals_module):
    vocal = SimpleNamespace(cnn_mask=None)
    list_of_vocals = make_list_of_vocals(list_of_vocals_module, [vocal])
    mask = np.ones((1, 2, 3), dtype=np.uint8) * 255

    list_of_vocals.add_segmentation_masks_to_vocals(mask)

    assert np.array_equal(vocal.cnn_mask, mask[0])
    assert list_of_vocals.has_cnn_masks() is True


def test_remove_visual_data_clears_all_transient_images(list_of_vocals_module):
    vocal = SimpleNamespace(spectrogram=np.ones((2, 2)), mask=np.ones((2, 2)), cnn_mask=np.ones((2, 2)))
    list_of_vocals = make_list_of_vocals(list_of_vocals_module, [vocal])

    list_of_vocals.remove_visual_data()

    assert vocal.spectrogram is None
    assert vocal.mask is None
    assert vocal.cnn_mask is None


def test_combine_list_of_list_of_vocals_handles_all_empty_inputs(list_of_vocals_module):
    list_of_vocals = list_of_vocals_module.ListOfVocals()
    empty_segment = list_of_vocals_module.ListOfVocals()

    list_of_vocals.combine_list_of_list_of_vocals([empty_segment])

    assert list_of_vocals.number_of_vocals == 0
    assert list_of_vocals.vocals_combined is True
