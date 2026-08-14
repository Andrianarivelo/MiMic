# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import numpy as np

from vocalpy.pipelines.animal import Animal


class Guineapig(Animal):
    def classify_vocalizations(self, network_type, list_of_vocals, source=None):
        raise NotImplementedError("classify_vocalizations not implemented for guinea pigs")

    def get_spectrogram_kwargs(self):
        kwargs = super().get_spectrogram_kwargs()
        kwargs.update({"window_type": "barthann", "window_size": 512, "noverlap": 256, "nfft": 1024})
        return kwargs

    def get_component_min_area(self):
        return 100

    def get_background_window_radius(self):
        return 25

    def get_output_spectrogram_range(self):
        return 100

    def get_median_filter_size(self):
        return (4, 4)

    def adjust_normalized_spectrogram(self, normalized_spectrogram):
        lower_percentile, _ = np.percentile(normalized_spectrogram, (1, 99))
        adjusted = normalized_spectrogram.copy()
        adjusted[adjusted < lower_percentile] = 0
        adjusted[adjusted > lower_percentile] = 1
        return adjusted

    def apply_morphology(self, binary_spectrogram):
        return binary_spectrogram

    def finalize_candidate_mask(self, candidate_mask):
        return candidate_mask.astype(np.uint8)

    def check_if_vocals_are_close(self, base_vocal, next_vocal):
        # -- conditions to check:
        # -- 1) next vocal starts within 100ms from base vocal start time AND
        # -- next vocal frequency is higher than the base vocal frequency
        # -- 2) next vocal starts within 100ms from base vocal end time AND
        # -- next vocal frequency is higher than the base vocal frequency
        # -- 3) next vocal starts/ends within base vocal start/end (harmonic)
        # -- 4) duration limit

        max_interval = 0.1  # 100ms
        condition_1 = (np.abs(base_vocal.end - next_vocal.start) < max_interval) and (
            next_vocal.min_freq > base_vocal.max_freq
        )
        condition_2 = (np.abs(base_vocal.start - next_vocal.start) < max_interval) and (
            next_vocal.min_freq > base_vocal.max_freq
        )
        condition_3 = next_vocal.start >= base_vocal.start and next_vocal.end <= base_vocal.end
        condition_4 = base_vocal.duration <= 0.25

        return True if condition_4 and (condition_1 or condition_2 or condition_3) else False
