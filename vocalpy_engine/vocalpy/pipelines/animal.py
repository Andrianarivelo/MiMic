# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

from abc import ABC, abstractmethod

import cv2
import numpy as np

from time import time
from scipy import ndimage
from logging import getLogger
from skimage import exposure, measure

from vocalpy.utils.io import read_audio
from vocalpy.nn.classifier import VocalClassifier
from vocalpy.nn.segmenter import VocalSegmenter
from vocalpy.nn.datasets import create_array_from_list_of_vocals
from vocalpy.nn.pretrained_models import get_pretrained_model_spec
from vocalpy.utils.signal_processing import compute_spectrogram
from vocalpy.utils.image_processing import normalize, contrast_adjustment, bradley_roth


class Animal(ABC):
    """
    Animal class calls apropriate pipeline functions
    """

    def __init__(self, animal, params):
        self.params = params
        self._animal = animal

    def identify_vocalizations(self, chunk):
        return self.identifier(chunk)

    def classify_vocalizations(self, network_type, list_of_vocals, source=None):
        if list_of_vocals.number_of_vocals == 0:
            classes = list(get_pretrained_model_spec(network_type).classes)
            if network_type == "noise":
                return np.asarray([], dtype=bool), classes
            return np.empty((0, len(classes)), dtype=float), classes

        source = create_array_from_list_of_vocals(list_of_vocals) if source is None else source
        classifier = VocalClassifier(network_type=network_type, source=source)
        predictions = classifier.classify_list_of_vocals(list_of_vocals)
        return predictions, classifier.classes

    def segment_vocalizations(self, list_of_vocals, source=None, path_to_model=None, threshold=None):
        if list_of_vocals.number_of_vocals == 0:
            return np.empty((0, 0, 0), dtype=np.uint8)

        source = create_array_from_list_of_vocals(list_of_vocals) if source is None else source
        segmenter = VocalSegmenter(
            source=source,
            path_to_model=path_to_model,
            threshold=threshold,
        )
        return segmenter.segment_list_of_vocals(list_of_vocals)

    def parse_chunk(self, chunk):
        """
        Convert the mixed-type chunk payload back into typed values.
        """
        (
            audio_path,
            output_dir,
            spectrogram_dir,
            mask_dir,
            sample_rate,
            bin_size,
            this_bin,
            start_range,
            end_range,
        ) = chunk

        return (
            audio_path,
            output_dir,
            spectrogram_dir,
            mask_dir,
            int(float(sample_rate)),
            int(float(bin_size)),
            int(float(this_bin)),
            int(float(start_range)),
            int(float(end_range)) if end_range is not None else None,
        )

    def get_centered_window_bounds(self, center, window_radius, max_width):
        """
        Clamp a centered window to valid bounds while preserving width when possible.
        """
        target_width = min(max_width, window_radius * 2)
        if max_width <= target_width:
            return 0, max_width

        start = center - window_radius
        end = center + window_radius

        if start < 0:
            end = min(max_width, end - start)
            start = 0

        if end > max_width:
            start = max(0, start - (end - max_width))
            end = max_width

        return int(start), int(end)

    def estimate_background_intensity(self, spectrogram, center, window_radius):
        """
        Estimate local background intensity around a candidate vocalization.
        """
        start, end = self.get_centered_window_bounds(center, window_radius, spectrogram.shape[1])
        return float(np.mean(spectrogram[:, start:end]))

    def has_minimum_contrast(self, signal_intensity, background_intensity):
        """
        Compare signal and background in dB space.
        """
        min_contrast_db = self.params.get("min_contrast_db", 3.0)
        return (signal_intensity - background_intensity) >= min_contrast_db

    def get_region_intensity_stats(self, prop):
        """
        Read intensity statistics across skimage regionprops API versions.
        """
        min_intensity = prop.intensity_min if hasattr(prop, "intensity_min") else prop.min_intensity
        max_intensity = prop.intensity_max if hasattr(prop, "intensity_max") else prop.max_intensity
        mean_intensity = prop.intensity_mean if hasattr(prop, "intensity_mean") else prop.mean_intensity
        return min_intensity, max_intensity, mean_intensity

    def get_duration_limits_in_frames(self, time_resolution_ms):
        """
        Resolve duration thresholds to spectrogram-frame counts.

        Legacy configs used `min_vocal_duration` / `max_vocal_duration` as
        frame counts despite the ambiguous name. Preserve that behavior while
        allowing explicit frame or millisecond keys going forward.
        """
        if "min_vocal_duration_ms" in self.params:
            min_duration = int(np.ceil(self.params["min_vocal_duration_ms"] / time_resolution_ms))
        elif "min_vocal_duration_frames" in self.params:
            min_duration = int(self.params["min_vocal_duration_frames"])
        else:
            min_duration = int(self.params["min_vocal_duration"])

        if "max_vocal_duration_ms" in self.params:
            max_duration = int(np.floor(self.params["max_vocal_duration_ms"] / time_resolution_ms))
        elif "max_vocal_duration_frames" in self.params:
            max_duration = int(self.params["max_vocal_duration_frames"])
        else:
            max_duration = int(self.params["max_vocal_duration"])

        return min_duration, max_duration

    def get_spectrogram_kwargs(self):
        return {
            "window_type": self.params["window_type"],
            "window_size": self.params["window_size"],
            "noverlap": self.params["noverlap"],
            "nfft": self.params["nfft"],
            "lower_frequency_cutoff": self.params["lower_frequency_cutoff"],
            "higher_frequency_cutoff": self.params["higher_frequency_cutoff"],
        }

    def get_component_min_area(self):
        return 20

    def get_background_window_radius(self):
        return 200

    def get_output_spectrogram_range(self):
        return 206

    def adjust_normalized_spectrogram(self, normalized_spectrogram):
        return contrast_adjustment(data=normalized_spectrogram, lower_percentile=1, upper_percentile=99)

    def get_median_filter_size(self):
        return (3, 3)

    def apply_morphology(self, binary_spectrogram):
        kernel_rect = np.ones((4, 2), np.uint8)
        kernel_line1 = np.ones((4, 1), np.uint8)
        kernel_line2 = np.ones((5, 1), np.uint8)

        binary_spectrogram = cv2.erode(binary_spectrogram, kernel_line1, iterations=1)
        binary_spectrogram = cv2.dilate(binary_spectrogram, kernel_rect, iterations=1)
        binary_spectrogram = cv2.dilate(binary_spectrogram, kernel_line2, iterations=1)
        return cv2.erode(binary_spectrogram, kernel_line1, iterations=2)

    def finalize_candidate_mask(self, candidate_mask):
        kernel_line3 = np.ones((1, 3), dtype=np.uint8)
        return cv2.dilate(candidate_mask.astype(np.uint8), kernel_line3, iterations=1)

    def build_candidate_mask(self, spectrogram):
        normalized = normalize(data=spectrogram)
        adjusted = self.adjust_normalized_spectrogram(normalized)
        binary = bradley_roth(adjusted, t=20)
        filtered = ndimage.median_filter(binary, size=self.get_median_filter_size())
        morphed = self.apply_morphology(filtered)
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(morphed, 4, cv2.CV_32S)

        areas = stats[1:, 4]
        candidate_mask = np.zeros(labels.shape, dtype=np.uint8)
        min_area = self.get_component_min_area()
        for index in range(num_components - 1):
            if areas[index] >= min_area:
                candidate_mask[labels == index + 1] = 255

        return self.finalize_candidate_mask(candidate_mask)

    def get_sorted_regionprops(self, candidate_mask, spectrogram):
        labels = measure.label(candidate_mask, background=0)
        props = measure.regionprops(labels, intensity_image=spectrogram, cache=True)
        return sorted(props, key=lambda prop: np.min(prop.coords[:, 1]), reverse=False)

    def get_time_range_label(self, sample_range, sample_rate, start_range, end_range):
        if end_range is None:
            return (
                f"time range: {sample_range.shape[0] / sample_rate:.2f}s; "
                f"audio range: {start_range / sample_rate:.2f}s-end of audio"
            )
        return (
            f"time range: {sample_range.shape[0] / sample_rate:.2f}s; "
            f"audio range: {start_range / sample_rate:.2f}-{end_range / sample_rate:.2f}s"
        )

    def get_vocal_times(self, start, end, time_res, this_bin, bin_size):
        base_offset = (this_bin - 1) * bin_size
        if this_bin == 1:
            base_offset += 0.5
        return (start * time_res) + base_offset, (end * time_res) + base_offset

    def build_vocal_from_prop(self, prop, spectrogram, time_res, freq_res, this_bin, bin_size, lower_frequency_cutoff):
        from vocalpy.modules.vocal import Vocal

        min_intensity, max_intensity, mean_intensity = self.get_region_intensity_stats(prop)
        start = int(np.min(prop.coords[:, 1]))
        end = int(np.max(prop.coords[:, 1]))
        centroid_time = int(np.ceil(prop.centroid[1]))
        background_intensity = self.estimate_background_intensity(spectrogram, centroid_time, self.get_background_window_radius())
        if not self.has_minimum_contrast(mean_intensity, background_intensity):
            return None

        min_freq_coord = int(np.min(prop.coords[:, 0]))
        max_freq_coord = int(np.max(prop.coords[:, 0]))
        start_time, end_time = self.get_vocal_times(start, end, time_res, this_bin, bin_size)
        min_freq = (min_freq_coord * freq_res) + lower_frequency_cutoff
        max_freq = (max_freq_coord * freq_res) + lower_frequency_cutoff

        return Vocal(
            bin_number=this_bin,
            start=start_time,
            end=end_time,
            start_coord=start,
            end_coord=end,
            duration=(end - start) * time_res * 1000,
            interval=0,
            min_freq=min_freq,
            max_freq=max_freq,
            min_freq_coord=min_freq_coord,
            max_freq_coord=max_freq_coord,
            avg_freq=(np.mean(prop.coords[:, 0]) * freq_res) + lower_frequency_cutoff,
            bandwidth=max_freq - min_freq,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
            avg_intensity=mean_intensity,
            bg_intensity=background_intensity,
            area=prop.area,
            centroid=np.rint(prop.centroid).astype(int),
            coords=prop.coords,
        )

    def create_list_of_vocals(self, props, spectrogram, candidate_mask, time_res, freq_res, this_bin, bin_size):
        from vocalpy.modules.list_of_vocals import ListOfVocals

        min_duration_frames, max_duration_frames = self.get_duration_limits_in_frames(time_res * 1000)
        vocals = []
        previous_end = None

        for prop in props:
            start = int(np.min(prop.coords[:, 1]))
            end = int(np.max(prop.coords[:, 1]))
            duration = end - start
            if duration < min_duration_frames or duration > max_duration_frames:
                continue

            vocal = self.build_vocal_from_prop(
                prop=prop,
                spectrogram=spectrogram,
                time_res=time_res,
                freq_res=freq_res,
                this_bin=this_bin,
                bin_size=bin_size,
                lower_frequency_cutoff=self.params["lower_frequency_cutoff"],
            )
            if vocal is None:
                continue

            vocal.interval = 0 if previous_end is None else abs(previous_end - start) * time_res
            vocals.append(vocal)
            previous_end = end

        list_of_vocals = ListOfVocals(vocals_in_recording=np.asarray(vocals, dtype=object))
        if list_of_vocals.number_of_vocals > 0:
            self.connect_vocals(list_of_vocals)
            list_of_vocals.update_centroids()

            spectrogram_range = self.get_output_spectrogram_range()
            list_of_vocals.update_coords(spectrogram_range)
            scaled_spectrogram = exposure.rescale_intensity(spectrogram, in_range="image", out_range=np.uint8)
            list_of_vocals.add_spectrograms_to_vocals(
                full_spectrogram=np.flipud(scaled_spectrogram),
                full_mask=np.flipud(candidate_mask),
                spec_range=spectrogram_range,
            )
        return list_of_vocals

    def identifier(self, chunk):
        logger = getLogger()
        time_bin_start = time()

        (
            audio_path,
            _output_dir,
            _spectrogram_dir,
            _mask_dir,
            sample_rate,
            bin_size,
            this_bin,
            start_range,
            end_range,
        ) = self.parse_chunk(chunk)

        time_audio_read = time()
        logger.info(f"[bin {this_bin}]: reading audio;")
        sample_range, __ = read_audio(audio_path, start=start_range, stop=end_range)
        logger.info(f"[bin {this_bin}]: read audio runtime: {time() - time_audio_read:.2f}s;")

        time_spectrogram = time()
        logger.info(
            f"[bin {this_bin}]: computing spectrogram; "
            f"{self.get_time_range_label(sample_range, sample_rate, start_range, end_range)}"
        )
        frequencies, times, spectrogram = compute_spectrogram(
            samples=sample_range,
            fs=sample_rate,
            **self.get_spectrogram_kwargs(),
        )
        time_res = (sample_range.shape[0] / sample_rate) / times.shape[0]
        freq_res = (np.max(frequencies) - self.params["lower_frequency_cutoff"]) / frequencies.shape[0]
        logger.info(f"[bin {this_bin}]: spectrogram runtime: {time() - time_spectrogram:.2f}s")
        logger.info(f"[bin {this_bin}]: time resolution: {time_res * 1000:.2f}ms")
        logger.info(f"[bin {this_bin}]: freq resolution: {freq_res:.2f}Hz")

        time_mask = time()
        candidate_mask = self.build_candidate_mask(spectrogram)
        logger.info(f"[bin {this_bin}]: candidate mask runtime: {time() - time_mask:.2f}s")

        time_region_props = time()
        props = self.get_sorted_regionprops(candidate_mask, spectrogram)
        logger.info(f"[bin {this_bin}]: region props runtime: {time() - time_region_props:.2f}s")

        time_vocals = time()
        list_of_vocals = self.create_list_of_vocals(
            props,
            spectrogram,
            candidate_mask,
            time_res,
            freq_res,
            this_bin,
            bin_size,
        )
        raw_count = list_of_vocals.number_of_vocals
        logger.info(f"[bin {this_bin}]: list of vocals runtime: {time() - time_vocals:.2f}s")
        logger.info(f"[bin {this_bin}]: raw number of vocals: {raw_count}")
        logger.info(f"[bin {this_bin}]: {list_of_vocals}")
        logger.info(f"[bin {this_bin}]: bin runtime: {time() - time_bin_start:.2f}s")

        return list_of_vocals

    @abstractmethod
    def check_if_vocals_are_close(self, first_vocal, second_vocal):
        return NotImplemented

    def connect_vocals(self, list_of_vocals):
        """
        Checks segmentation and combines segments that belong to the same vocalization. Uses predefined
        constraints. Can have different contrainsts for different animal pipelines

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`
            list of vocal candidates to be connected
        """
        vocals = [] if list_of_vocals.vocals_in_recording is None else list(list_of_vocals.vocals_in_recording)
        if not vocals:
            list_of_vocals.vocals_in_recording = []
            list_of_vocals.number_of_vocals = 0
            list_of_vocals.vocals_combined = True
            return 0

        merged_vocals = []
        current_vocal = vocals[0]

        for next_vocal in vocals[1:]:
            if self.check_if_vocals_are_close(current_vocal, next_vocal):
                current_vocal = self.combine_vocals(current_vocal, next_vocal)
            else:
                merged_vocals.append(current_vocal)
                current_vocal = next_vocal

        merged_vocals.append(current_vocal)
        list_of_vocals.vocals_in_recording = np.asarray(merged_vocals, dtype=object)
        list_of_vocals.number_of_vocals = len(merged_vocals)
        list_of_vocals.vocals_combined = True
        return 0

    def combine_vocals(self, first_vocal, second_vocal):
        """
        Combines two vocals

        Parameters
        ----------
        first_vocal : :class:`Vocal`
        second_vocal : :class:`Vocal`
            vocals to be combined
        """
        from vocalpy.modules.vocal import Vocal

        start_difference = first_vocal.start - second_vocal.start
        end_difference = first_vocal.end - second_vocal.end

        combined_vocal = Vocal(
            bin_number=first_vocal.bin_number
            if (first_vocal.bin_number < second_vocal.bin_number)
            else second_vocal.bin_number,
            start=first_vocal.start if (start_difference < 0) else second_vocal.start,
            start_coord=first_vocal.start_coord if (start_difference < 0) else second_vocal.start_coord,
            end=first_vocal.end if (end_difference > 0) else second_vocal.end,
            end_coord=first_vocal.end_coord if (end_difference > 0) else second_vocal.end_coord,
            interval=-1,
            min_freq=first_vocal.min_freq if (first_vocal.min_freq < second_vocal.min_freq) else second_vocal.min_freq,
            max_freq=first_vocal.max_freq if (first_vocal.max_freq > second_vocal.max_freq) else second_vocal.max_freq,
            min_freq_coord=first_vocal.min_freq_coord
            if (first_vocal.min_freq_coord < second_vocal.min_freq_coord)
            else second_vocal.min_freq_coord,
            max_freq_coord=first_vocal.max_freq_coord
            if (first_vocal.max_freq_coord > second_vocal.max_freq_coord)
            else second_vocal.max_freq_coord,
            avg_freq=np.mean((first_vocal.avg_freq, second_vocal.avg_freq)),
            min_intensity=first_vocal.min_intensity
            if (first_vocal.min_intensity < second_vocal.min_intensity)
            else second_vocal.min_intensity,
            max_intensity=first_vocal.max_intensity
            if (first_vocal.max_intensity < second_vocal.max_intensity)
            else second_vocal.max_intensity,
            avg_intensity=np.mean((first_vocal.avg_intensity, second_vocal.avg_intensity)),
            bg_intensity=np.mean((first_vocal.bg_intensity, second_vocal.bg_intensity)),
            area=first_vocal.area + second_vocal.area,
            centroid=first_vocal.centroid,
            coords=np.vstack((first_vocal.coords, second_vocal.coords)),
        )
        combined_vocal.duration = (combined_vocal.end - combined_vocal.start) * 1000
        combined_vocal.bandwidth = combined_vocal.max_freq - combined_vocal.min_freq

        return combined_vocal
