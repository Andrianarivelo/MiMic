# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import torch

import numpy as np

from vocalpy.utils.io import write_pickle_file


class ListOfVocals(object):
    """
    List of vocalizations identified in the recording. Each vocal if an instance of :class:`Vocals`

    Parameters
    ----------
    vocals_in_recording : List[:class:`Vocal`]
        list of vocals identified in the recording
    """

    def __init__(self, vocals_in_recording=None):
        self.vocals_in_recording = self._normalize_vocals(vocals_in_recording)
        self.number_of_vocals = len(self.vocals_in_recording)

        self.vocals_combined = False
        self.intervals_fixed = False
        self.centroid_spectro_fixed = False
        self.coords_fixed = False

    @staticmethod
    def _normalize_vocals(vocals_in_recording):
        if vocals_in_recording is None:
            return np.asarray([], dtype=object)

        normalized_vocals = np.asarray(vocals_in_recording, dtype=object)
        if normalized_vocals.size == 0:
            return np.asarray([], dtype=object)
        return normalized_vocals.reshape(-1)

    def __str__(self):
        return f"{self.__class__.__name__}:\n \
            number_of_vocals: {self.number_of_vocals}\n \
            vocals_combined: {self.vocals_combined}\n \
            intervals_fixed: {self.intervals_fixed}\n \
            centroid_spectro_fixed: {self.centroid_spectro_fixed}"

    def save_list_of_vocals_object(self, path):
        """
        Saves a :class:`ListOfVocals` Object to file

        Parameters
        ----------
        path : str
            path to save the object
        """
        write_pickle_file(self, "list_of_vocals", path, object_type="list_of_vocals")

    def update_intervals(self):
        """
        Updates the interval (silence) between vocals. Usually used after combining or removing vocals
        """
        # -- go through vocals and update inter vocal times
        if self.number_of_vocals == 0:
            self.intervals_fixed = True
            return 0

        self.vocals_in_recording[0].interval = 0

        for idx in range(1, len(self.vocals_in_recording), 1):
            self.vocals_in_recording[idx].interval = np.abs(
                (self.vocals_in_recording[idx - 1].end - self.vocals_in_recording[idx].start)
            )

        self.intervals_fixed = True
        return 0

    def update_centroids(self):
        """
        Updates centroid coordinates for each vocal. Usually called after combining vocals. Absolute
        centroid from vocal coordinates start/end, and min/max frequency
        """
        for vocal in self.vocals_in_recording:
            cx = vocal.start_coord + ((vocal.end_coord - vocal.start_coord) // 2)
            cy = vocal.min_freq_coord + ((vocal.max_freq_coord - vocal.min_freq_coord) // 2)
            vocal.centroid = np.rint([cy, cx]).astype(int)

        self.centroid_spectro_fixed = True
        return 0

    def update_coords(self, spec_range=200):
        """
        Updates coordinates for each vocal after cropping area around a vocal. Absolute
        coords from vocal coordinates start/end, and min/max frequency

        Parameters
        ----------
        spec_range : int, optional
            range before/after vocal used to crop and generate spectrograms
        """
        for vocal in self.vocals_in_recording:
            col_values = vocal.coords[:, 1]
            # make column values zero-centered by subtracting the mean
            col_values = col_values - int(np.mean(col_values))
            # col values will be centered in the spectrogram
            col_values = col_values + spec_range
            vocal.coords[:, 1] = col_values

        self.coords_fixed = True
        return 0

    def combine_list_of_list_of_vocals(self, list_of_list_of_vocals):
        """
        Combines a list of :class:`ListOfVocals` into one :class:`ListOfVocals`. Usually called
        to combine several lists of parallel processing of a recording

        Parameters
        ----------
        list_of_list_of_vocals : List[:class:`ListOfVocals`]
            list of :class:`ListOfVocals`
        """
        new_list_of_vocals = []
        for list_of_vocals in list_of_list_of_vocals:
            if list_of_vocals is None:
                continue
            normalized_vocals = self._normalize_vocals(list_of_vocals.vocals_in_recording)
            if normalized_vocals.size > 0:
                new_list_of_vocals.append(normalized_vocals)

        self.vocals_in_recording = (
            np.hstack(new_list_of_vocals) if new_list_of_vocals else np.asarray([], dtype=object)
        )
        self.number_of_vocals = len(self.vocals_in_recording)
        self.vocals_combined = True
        self.centroid_spectro_fixed = True
        return 0

    def add_spectrograms_to_vocals(self, full_spectrogram, full_mask, spec_range=200):
        """
        Stores the spectrogram in each :class:`Vocal` class object in the :class:`ListOfVocals`

        Parameters
        ----------
        full_spectrogram : ndarray
            complete spectrogram ranging the recording segment
        full_mask : ndarray
            complete segmentation mask ranging the recording segment
        spec_range : int, optional
            range to crop the spectrogram/segmentation around the vocal (+-200)
        """
        for vocal in self.vocals_in_recording:
            cy, cx = vocal.centroid
            vocal.spectrogram = self._centered_crop(full_spectrogram, cx, spec_range)
            vocal.mask = self._centered_crop(full_mask, cx, spec_range)
            vocal.centroid = [vocal.centroid[0], spec_range]
            self.centroid_spectro_fixed = True
        return 0

    @staticmethod
    def _centered_crop(data, center, spec_range):
        """
        Return a fixed-width crop padded with zeros when the source hits an edge.
        """
        target_width = spec_range * 2
        source_start = center - spec_range
        source_end = center + spec_range

        crop_start = max(0, source_start)
        crop_end = min(data.shape[1], source_end)
        cropped = data[:, crop_start:crop_end]

        if cropped.shape[1] == target_width:
            return cropped

        output = np.zeros((data.shape[0], target_width), dtype=data.dtype)
        insert_start = max(0, -source_start)
        insert_end = insert_start + cropped.shape[1]
        output[:, insert_start:insert_end] = cropped
        return output

    def save_spectrograms(self, output_dir=None):
        """
        Saves the spectrogram image to the output directory

        Parameters
        ----------
        output_dir : str, optional
            path to output directory to save the files
        """
        for filename, vocal in enumerate(self.vocals_in_recording, start=1):
            vocal.save_spectrogram_as_image(path=output_dir, filename=str(filename))
        return 0

    def save_validation_images(self, output_dir=None):
        """
        Saves the spectrogram overlaidd with the segmentation image to the output directory

        Parameters
        ----------
        output_dir : str, optional
            path to output directory to save the files
        """
        for filename, vocal in enumerate(self.vocals_in_recording, start=1):
            vocal.save_spectrogram_with_segmentation_as_image(path=output_dir, filename=str(filename))
        return 0

    def save_masks(self, output_dir=None):
        """
        Saves the segmentation mask image to the output directory

        Parameters
        ----------
        output_dir : str, optional
            path to output directory to save the files
        """
        for filename, vocal in enumerate(self.vocals_in_recording, start=1):
            vocal.save_mask_as_image(path=output_dir, filename=str(filename))
        return 0

    def save_cnn_masks(self, output_dir=None):
        """
        Saves the neural segmentation mask image to the output directory

        Parameters
        ----------
        output_dir : str, optional
            path to output directory to save the files
        """
        for filename, vocal in enumerate(self.vocals_in_recording, start=1):
            if vocal.cnn_mask is None:
                continue
            vocal.save_cnn_mask_as_image(path=output_dir, filename=str(filename))
        return 0

    def remove_spectrograms(self):
        """
        Removes the spectrogram data from each :class:`Vocal` in the :class:`ListOfVocals`
        """
        for vocal in self.vocals_in_recording:
            vocal.spectrogram = None
        return 0

    def remove_masks(self):
        """
        Removes the segmentation data from each :class:`Vocal` in the :class:`ListOfVocals`
        """
        for vocal in self.vocals_in_recording:
            vocal.mask = None
        return 0

    def remove_cnn_masks(self):
        """
        Removes the neural segmentation data from each :class:`Vocal` in the :class:`ListOfVocals`
        """
        for vocal in self.vocals_in_recording:
            vocal.cnn_mask = None
        return 0

    def remove_visual_data(self):
        """
        Removes transient image payloads from each :class:`Vocal`.
        """
        self.remove_spectrograms()
        self.remove_masks()
        self.remove_cnn_masks()
        return 0

    def add_segmentation_masks_to_vocals(self, masks):
        """
        Updates :class:`ListOfVocals` with neural-network segmentation masks.

        Parameters
        ----------
        masks : numpy.ndarray
            binary mask predictions shaped (N, H, W)
        """
        masks = np.asarray(masks)
        if masks.shape[0] != self.number_of_vocals:
            raise ValueError(
                "number of vocals and segmentation masks differ. "
                f"number of vocals: {self.number_of_vocals}; "
                f"number of masks: {masks.shape[0]}"
            )

        for idx, vocal in enumerate(self.vocals_in_recording):
            vocal.cnn_mask = masks[idx]
        return 0

    def has_cnn_masks(self):
        return any(vocal.cnn_mask is not None for vocal in self.vocals_in_recording)

    def remove_vocals_classified_as_noise(self, predictions):
        """
        Removes vocals that were classified as noise from the :class:`ListOfVocals` and
        updates the number of vocals

        Parameters
        ----------
        predictions : List[float]
            Neural Network classification predictions for the :class:`ListOfVocals`
        """
        self.vocals_in_recording = self.vocals_in_recording[predictions]
        self.number_of_vocals = len(self.vocals_in_recording)
        if self.number_of_vocals > 0:
            self.update_intervals()
        return 0

    def add_classification_to_vocals(self, predictions, classes):
        """
        Updates :class:`ListOfVocals` with the class and probability distribution obtained
        using the Neural Network

        Parameters
        ----------
        predictions : List[float]
            Neural Network classification predictions for the :class:`ListOfVocals`
        classes : List[str]
            Labels used for classifying vocalizations
        """
        for idx, vocal in enumerate(self.vocals_in_recording):
            vocal.probabilities = predictions[idx]

            # -- convert numpy to torch, to use TopK function
            preds = torch.tensor(predictions[idx])
            # -- get top2 probabilities and their class names
            top1, top2 = preds.topk(2).indices.numpy()
            vocal.top1 = classes[top1]
            vocal.top2 = classes[top2]

        return 0
