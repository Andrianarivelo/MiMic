# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import numpy as np

from vocalpy.pipelines.animal import Animal


class Rat(Animal):
    def get_spectrogram_kwargs(self):
        kwargs = super().get_spectrogram_kwargs()
        kwargs.update({"window_type": "hamming", "window_size": 256, "noverlap": 128, "nfft": 1024})
        return kwargs

    def get_output_spectrogram_range(self):
        return 400

    def check_if_vocals_are_close(self, first_vocal, second_vocal):
        # -- conditions to check:
        # -- 1) next vocal starts within 20ms from base vocal start time
        # -- 2) next vocal starts within 20ms from base vocal end time
        # -- 3) next vocal starts within base vocal start/end (harmonic)
        max_interval = 0.019  # 20ms - 1ms error because morph ops increase area
        condition_1 = np.abs(first_vocal.end - second_vocal.start) < max_interval
        condition_2 = np.abs(first_vocal.start - second_vocal.start) < max_interval
        condition_3 = (second_vocal.start >= first_vocal.start) and (second_vocal.start <= first_vocal.end)

        return True if (condition_1 or condition_2 or condition_3) else False
