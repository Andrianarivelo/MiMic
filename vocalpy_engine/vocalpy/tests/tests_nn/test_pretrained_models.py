# -*- coding: utf-8 -*-
"""Smoke tests for bundled pretrained models."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import torch

from vocalpy.nn.classifier import VocalClassifier
from vocalpy.nn.segmenter import VocalSegmenter
from vocalpy.nn.pretrained_models import PRETRAINED_MODEL_SPECS, validate_pretrained_model_file


def test_bundled_checkpoint_hashes_match_expected_metadata():
    for spec in PRETRAINED_MODEL_SPECS.values():
        assert validate_pretrained_model_file(spec.path, expected_sha256=spec.sha256) == spec.sha256
        assert spec.metadata_path.exists() is True
        if spec.network_type == "segment":
            assert spec.input_shape == (1, 512, 512)
            assert spec.prediction_type == "logits"
            assert spec.default_threshold == 0.51
        else:
            assert spec.input_shape == (3, 224, 224)


def test_bundled_noise_checkpoint_loads_and_runs_on_cpu():
    classifier = VocalClassifier.__new__(VocalClassifier)

    model = classifier.load_pretrained_noise_model(torch.device("cpu"))
    dummy_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

    output_a = model(dummy_input)
    output_b = model(dummy_input)

    assert tuple(output_a.shape) == (1, 2)
    assert torch.isfinite(output_a).all()
    torch.testing.assert_close(output_a, output_b)
    assert classifier.classes == list(PRETRAINED_MODEL_SPECS["noise"].classes)


def test_bundled_class_checkpoint_loads_and_runs_on_cpu():
    classifier = VocalClassifier.__new__(VocalClassifier)

    model = classifier.load_pretrained_class_model(torch.device("cpu"))
    dummy_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

    output_a = model(dummy_input)
    output_b = model(dummy_input)

    assert tuple(output_a.shape) == (1, 11)
    assert torch.isfinite(output_a).all()
    torch.testing.assert_close(output_a, output_b)
    assert classifier.classes == list(PRETRAINED_MODEL_SPECS["class"].classes)


def test_bundled_segment_checkpoint_loads_and_runs_on_cpu():
    segmenter = VocalSegmenter.__new__(VocalSegmenter)

    model = segmenter.load_segmentation_model(torch.device("cpu"))
    dummy_input = torch.zeros((1, 1, 512, 512), dtype=torch.float32)

    output_a = model(dummy_input)
    output_b = model(dummy_input)

    assert tuple(output_a.shape) == (1, 1, 512, 512)
    assert torch.isfinite(output_a).all()
    torch.testing.assert_close(output_a, output_b)
