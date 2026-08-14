# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"


import numpy as np

from math import ceil

from vocalpy.utils.io import read_audio_information


class Audio(object):
    """
    Audio object class
    Reads the audio file and breaks it down into segments of one minute to be processed
    in parallel. Stores metadata about the audio recording.

    Parameters
    ----------
    audio_path : str
        Path to audio file to be analyzed
    output_dir : str
        Path to write output files
    spectrogram_dir : str
        Path to write spectrograms
    mask_dir : str
        Path to write segmentation masks
    bin_size : int
        Bin size to partition audio file for parallel processing

    Returns
    -------
    audio : Object
        The audio object containing the read audio file and metadata
    """

    def __init__(self, audio_path, output_dir, spectrogram_dir, mask_dir, bin_size):
        self.audio_path = audio_path
        self.sampling_rate = None
        self.number_of_samples = None
        self.audio_duration = None
        self.read_audio_metadata()
        self.bin_size = bin_size if (bin_size < self.audio_duration) else self.audio_duration
        self.bins = ceil(self.audio_duration / self.bin_size)
        self.chunks = self.create_chunks(output_dir, spectrogram_dir, mask_dir)

    def read_audio_metadata(self):
        """
        Reads audio metadata. Stores information in the Recording Object
        """
        metadata = read_audio_information(self.audio_path)
        self.sampling_rate = metadata.samplerate
        self.audio_duration = metadata.duration
        self.number_of_samples = self.audio_duration * self.sampling_rate

    def create_chunks(self, output_dir, spectrogram_dir, mask_dir, overlap=0.15):
        """
        Segments audio for parallel or sequential processing

        Parameters
        ----------
            overlap : float
                segments overlap (in seconds)
        """
        chunks = []
        baseline_chunk = [
            self.audio_path,
            output_dir,
            spectrogram_dir,
            mask_dir,
            self.sampling_rate,
            self.bin_size,
        ]
        for this_bin in range(1, self.bins + 1):
            # -- first bin, remove first 0.5 second of recording (usually noisy)
            if this_bin == 1:
                start_range = ceil(0.5 * self.sampling_rate)
                end_range = ceil((self.bin_size * self.sampling_rate) + (overlap * self.sampling_rate))
                end_range = end_range if end_range < self.number_of_samples else self.number_of_samples
                this_chunk = baseline_chunk.copy()
                this_chunk.append((this_bin, start_range, end_range,))
                chunks.append(np.hstack(this_chunk))

            elif this_bin == self.bins:  # -- last bin
                start_range = ceil((this_bin - 1) * self.bin_size * self.sampling_rate)
                end_range = self.audio_duration * self.sampling_rate
                if end_range - start_range < self.sampling_rate:
                    continue  # less than 1s
                # -- None reads until the end of the audio
                # end_range = None
                this_chunk = baseline_chunk.copy()
                this_chunk.append((this_bin, start_range, end_range,))
                chunks.append(np.hstack(this_chunk))

            else:  # -- all other bins
                start_range = ceil((this_bin - 1) * self.bin_size * self.sampling_rate)
                end_range = ceil((this_bin * self.bin_size * self.sampling_rate) + (overlap * self.sampling_rate))
                end_range = end_range if end_range < self.number_of_samples else self.number_of_samples
                if end_range - start_range < self.sampling_rate:
                    continue  # less than 1s
                this_chunk = baseline_chunk.copy()
                this_chunk.append((this_bin, start_range, end_range,))
                chunks.append(np.hstack(this_chunk))

        return chunks
