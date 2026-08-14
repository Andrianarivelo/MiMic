# -*- coding: utf-8 -*-
"""Regression tests for shared animal pipeline behavior."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

from types import SimpleNamespace

import numpy as np

from vocalpy.pipelines.animal import Animal


class StubAnimal(Animal):
    def identify_vocalizations(self, chunk):
        return chunk

    def check_if_vocals_are_close(self, first_vocal, second_vocal):
        return second_vocal.start <= first_vocal.end + 0.6

    def combine_vocals(self, first_vocal, second_vocal):
        return SimpleNamespace(
            bin_number=min(first_vocal.bin_number, second_vocal.bin_number),
            start=min(first_vocal.start, second_vocal.start),
            start_coord=min(first_vocal.start_coord, second_vocal.start_coord),
            end=max(first_vocal.end, second_vocal.end),
            end_coord=max(first_vocal.end_coord, second_vocal.end_coord),
            interval=-1,
            min_freq=min(first_vocal.min_freq, second_vocal.min_freq),
            max_freq=max(first_vocal.max_freq, second_vocal.max_freq),
            min_freq_coord=min(first_vocal.min_freq_coord, second_vocal.min_freq_coord),
            max_freq_coord=max(first_vocal.max_freq_coord, second_vocal.max_freq_coord),
            avg_freq=(first_vocal.avg_freq + second_vocal.avg_freq) / 2,
            min_intensity=min(first_vocal.min_intensity, second_vocal.min_intensity),
            max_intensity=max(first_vocal.max_intensity, second_vocal.max_intensity),
            avg_intensity=(first_vocal.avg_intensity + second_vocal.avg_intensity) / 2,
            bg_intensity=(first_vocal.bg_intensity + second_vocal.bg_intensity) / 2,
            area=first_vocal.area + second_vocal.area,
            centroid=first_vocal.centroid,
            coords=list(first_vocal.coords) + list(second_vocal.coords),
            duration=(max(first_vocal.end, second_vocal.end) - min(first_vocal.start, second_vocal.start)) * 1000,
            bandwidth=max(first_vocal.max_freq, second_vocal.max_freq)
            - min(first_vocal.min_freq, second_vocal.min_freq),
        )


def make_vocal(start, end):
    return SimpleNamespace(
        bin_number=1,
        start=start,
        start_coord=start,
        end=end,
        end_coord=end,
        interval=0,
        min_freq=40,
        max_freq=80,
        min_freq_coord=1,
        max_freq_coord=2,
        avg_freq=60,
        min_intensity=-20,
        max_intensity=-10,
        avg_intensity=-15,
        bg_intensity=-30,
        area=1,
        centroid=(0, 0),
        coords=[(0, 0)],
    )


def make_list_of_vocals(vocals):
    return SimpleNamespace(
        vocals_in_recording=list(vocals),
        number_of_vocals=len(vocals),
        vocals_combined=False,
    )


def test_connect_vocals_handles_empty_input():
    animal = StubAnimal("mouse", {})
    list_of_vocals = make_list_of_vocals([])

    animal.connect_vocals(list_of_vocals)

    assert list_of_vocals.vocals_in_recording == []
    assert list_of_vocals.number_of_vocals == 0
    assert list_of_vocals.vocals_combined is True


def test_connect_vocals_merges_transitive_chain_without_duplicate_tail():
    animal = StubAnimal("mouse", {})
    list_of_vocals = make_list_of_vocals(
        [make_vocal(0.0, 1.0), make_vocal(1.4, 2.0), make_vocal(2.4, 3.0)]
    )

    animal.connect_vocals(list_of_vocals)

    merged_ranges = [(vocal.start, vocal.end) for vocal in list_of_vocals.vocals_in_recording]
    assert merged_ranges == [(0.0, 3.0)]
    assert list_of_vocals.number_of_vocals == 1


def test_background_window_stays_local_near_right_edge():
    animal = StubAnimal("mouse", {"min_contrast_db": 3.0})
    spectrogram = np.arange(60, dtype=float).reshape(3, 20)

    background = animal.estimate_background_intensity(spectrogram, center=19, window_radius=4)

    expected = np.mean(spectrogram[:, 12:20])
    assert background == expected


def test_minimum_contrast_uses_db_difference_not_ratio():
    animal = StubAnimal("mouse", {"min_contrast_db": 3.0})

    assert animal.has_minimum_contrast(-40.0, -50.0) is True
    assert animal.has_minimum_contrast(-48.5, -50.0) is False


def test_region_intensity_stats_supports_new_skimage_property_names():
    animal = StubAnimal("mouse", {})
    prop = SimpleNamespace(intensity_min=-90.0, intensity_max=-55.0, intensity_mean=-72.0)

    min_intensity, max_intensity, mean_intensity = animal.get_region_intensity_stats(prop)

    assert (min_intensity, max_intensity, mean_intensity) == (-90.0, -55.0, -72.0)


def test_duration_limits_preserve_legacy_frame_based_thresholds():
    animal = StubAnimal("mouse", {"min_vocal_duration": 5, "max_vocal_duration": 250})

    min_duration, max_duration = animal.get_duration_limits_in_frames(time_resolution_ms=0.512)

    assert (min_duration, max_duration) == (5, 250)


def test_duration_limits_can_be_expressed_explicitly_in_milliseconds():
    animal = StubAnimal("mouse", {"min_vocal_duration_ms": 5.0, "max_vocal_duration_ms": 250.0})

    min_duration, max_duration = animal.get_duration_limits_in_frames(time_resolution_ms=0.512)

    assert min_duration == 10
    assert max_duration == 488


def test_shared_classify_vocalizations_uses_common_classifier(monkeypatch):
    captured = {}

    class FakeClassifier:
        def __init__(self, network_type, source):
            captured["network_type"] = network_type
            captured["source"] = source
            self.classes = ["noise", "vocal"]

        def classify_list_of_vocals(self, list_of_vocals):
            captured["list_of_vocals"] = list_of_vocals
            return np.asarray([True, False])

    animal = StubAnimal("mouse", {})
    vocals = SimpleNamespace(number_of_vocals=2)

    monkeypatch.setattr("vocalpy.pipelines.animal.VocalClassifier", FakeClassifier)
    monkeypatch.setattr("vocalpy.pipelines.animal.create_array_from_list_of_vocals", lambda value: "derived-array")

    predictions, classes = animal.classify_vocalizations("noise", vocals)

    assert captured == {"network_type": "noise", "source": "derived-array", "list_of_vocals": vocals}
    assert predictions.tolist() == [True, False]
    assert classes == ["noise", "vocal"]


def test_shared_classify_vocalizations_short_circuits_empty_list(monkeypatch):
    animal = StubAnimal("mouse", {})
    vocals = SimpleNamespace(number_of_vocals=0)

    monkeypatch.setattr(
        "vocalpy.pipelines.animal.get_pretrained_model_spec",
        lambda network_type: SimpleNamespace(classes=("noise", "vocal")),
    )
    monkeypatch.setattr("vocalpy.pipelines.animal.VocalClassifier", lambda *args, **kwargs: None)

    predictions, classes = animal.classify_vocalizations("noise", vocals)

    assert predictions.tolist() == []
    assert classes == ["noise", "vocal"]


def test_shared_segment_vocalizations_uses_common_segmenter(monkeypatch):
    captured = {}

    class FakeSegmenter:
        def __init__(self, source, path_to_model, threshold):
            captured["source"] = source
            captured["path_to_model"] = path_to_model
            captured["threshold"] = threshold

        def segment_list_of_vocals(self, list_of_vocals):
            captured["list_of_vocals"] = list_of_vocals
            return np.ones((2, 3, 4), dtype=np.uint8) * 255

    animal = StubAnimal("mouse", {})
    vocals = SimpleNamespace(number_of_vocals=2)

    monkeypatch.setattr("vocalpy.pipelines.animal.VocalSegmenter", FakeSegmenter)
    monkeypatch.setattr("vocalpy.pipelines.animal.create_array_from_list_of_vocals", lambda value: "derived-array")

    predictions = animal.segment_vocalizations(vocals, path_to_model="/tmp/segmenter.pt", threshold=0.7)

    assert captured == {
        "source": "derived-array",
        "path_to_model": "/tmp/segmenter.pt",
        "threshold": 0.7,
        "list_of_vocals": vocals,
    }
    assert predictions.shape == (2, 3, 4)
