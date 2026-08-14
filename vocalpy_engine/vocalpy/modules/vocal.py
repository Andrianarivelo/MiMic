# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

from os.path import join
from vocalpy.utils.io import save_image_to_disk
from vocalpy.utils.image_processing import scatter_over_spectrogram, numpy_to_grayscale_image


class Vocal(object):
    """
    Vocalization object stores metadata abouteach vocal identified in the
    recording

    Parameters
    ----------
    bin_number : float, int
        bin number for this vocal (if recording uses 1 minute bins, bin
        indicates which minute the vocal happened)
    start : float, optional
        vocal start time in the recording
    end : float, optional
        vocal end time in the recording
    start_coord : float, optional
        vocal start coordinate in the recording spectrogram
    end_coord : float, optional
        vocal end coordinate in the recording spectrogram
    duration : float, optional
        vocal duration in seconds
    interval : float, optional
        distance from last vocal in seconds
    min_freq : float, optional
        vocal minimum frequency
    max_freq : float, optional
        vocal maximum frequency
    min_freq_coord : float, optional
        vocal minimum frequency coordinate in the spectrogram
    max_freq_coord : float, optional
        vocal maximum frequency coordinate in the spectrogram
    avg_freq : float, optional
        vocal average frequency
    bandwidth : float, optional
        vocal bandwidth (maximum frequency - minimum frequency)
    min_intensity : float, optional
        vocal minimum intensity in decibel
    max_intensity : float, optional
        vocal maximum intensity in decibel
    avg_intensity : float, optional
        vocal average intensity in decibel
    bg_intensity : float, optional
        background average intensity in decibel
    area : float, optional
        vocal area in pixels
    centroid : float, optional
        vocal centroid coordinates in the spectrogram
    orientation : float, optional
        vocal orientation (slope)
    coords : float, optional
        vocal coordinates in the spectrogram
    spectrogram : ndarray, optional
        spectrogram (image)
    mask : ndarray, optional
        segmentation (image)
    cnn_mask : ndarray, optional
        segmentation from neural network (image)
    probabilities : List[float], optional
        classes probability distribution from the neural network
    top1 : str, optional
        top 1 class from the neural network
    top2 : str, optional
        top 2 class from the neural network
    """

    def __init__(
        self,
        bin_number=None,
        start=None,
        end=None,
        start_coord=None,
        end_coord=None,
        duration=None,
        interval=None,
        min_freq=None,
        max_freq=None,
        min_freq_coord=None,
        max_freq_coord=None,
        avg_freq=None,
        bandwidth=None,
        min_intensity=None,
        max_intensity=None,
        avg_intensity=None,
        bg_intensity=None,
        area=None,
        centroid=None,
        orientation=None,
        coords=None,
        spectrogram=None,
        mask=None,
        cnn_mask=None,
        probabilities=None,
        top1=None,
        top2=None,
    ):

        self._bin_number = bin_number
        self._start = start
        self._end = end
        self._start_coord = start_coord
        self._end_coord = end_coord
        self._duration = duration
        self._interval = interval
        self._min_freq = min_freq
        self._max_freq = max_freq
        self._min_freq_coord = min_freq_coord
        self._max_freq_coord = max_freq_coord
        self._avg_freq = avg_freq
        self._bandwidth = bandwidth
        self._min_intensity = min_intensity
        self._max_intensity = max_intensity
        self._avg_intensity = avg_intensity
        self._bg_intensity = bg_intensity
        self._area = area
        self._centroid = centroid
        self._orientation = orientation
        self._coords = coords
        self._spectrogram = spectrogram
        self._mask = mask
        self._cnn_mask = cnn_mask
        self._probabilities = probabilities
        self._top1 = top1
        self._top2 = top2

    def __str__(self):
        return f"{self.__class__.__name__}\n \
            bin_number: {self.bin_number}\n \
            start: {self.start}\n \
            end: {self.end}\n \
            duration: {self.duration}\n \
            interval: {self.interval}\n \
            min_freq: {self.min_freq}\n \
            max_freq: {self.max_freq}\n \
            avg_freq: {self.avg_freq}\n \
            bandwidth: {self.bandwidth}\n \
            min_intensity: {self.min_intensity}\n \
            max_intensity: {self.max_intensity}\n \
            avg_intensity: {self.avg_intensity}\n \
            bg_intensity: {self.bg_intensity}\n \
            area: {self.area}\n \
            centroid: {self.centroid}\n \
            orientation: {self.orientation}\n"

    @property
    def bin_number(self):
        return self._bin_number

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def start_coord(self):
        return self._start_coord

    @property
    def end_coord(self):
        return self._end_coord

    @property
    def duration(self):
        return self._duration

    @property
    def interval(self):
        return self._interval

    @property
    def min_freq(self):
        return self._min_freq

    @property
    def max_freq(self):
        return self._max_freq

    @property
    def min_freq_coord(self):
        return self._min_freq_coord

    @property
    def max_freq_coord(self):
        return self._max_freq_coord

    @property
    def avg_freq(self):
        return self._avg_freq

    @property
    def bandwidth(self):
        return self._bandwidth

    @property
    def min_intensity(self):
        return self._min_intensity

    @property
    def max_intensity(self):
        return self._max_intensity

    @property
    def avg_intensity(self):
        return self._avg_intensity

    @property
    def bg_intensity(self):
        return self._bg_intensity

    @property
    def area(self):
        return self._area

    @property
    def centroid(self):
        return self._centroid

    @property
    def orientation(self):
        return self._orientation

    @property
    def coords(self):
        return self._coords

    @property
    def spectrogram(self):
        return self._spectrogram

    @property
    def mask(self):
        return self._mask

    @property
    def cnn_mask(self):
        return self._cnn_mask

    @property
    def probabilities(self):
        return self._probabilities

    @property
    def top1(self):
        return self._top1

    @property
    def top2(self):
        return self._top2

    @bin_number.setter
    def bin_number(self, new_bin_number):
        self._bin_number = new_bin_number

    @start.setter
    def start(self, new_start):
        self._start = new_start

    @end.setter
    def end(self, new_end):
        self._end = new_end

    @start_coord.setter
    def start_coord(self, new_start_coord):
        self._start_coord = new_start_coord

    @end_coord.setter
    def end_coord(self, new_end_coord):
        self._end_coord = new_end_coord

    @duration.setter
    def duration(self, new_duration):
        self._duration = new_duration

    @interval.setter
    def interval(self, new_interval):
        self._interval = new_interval

    @min_freq.setter
    def min_freq(self, new_min_freq):
        self._min_freq = new_min_freq

    @max_freq.setter
    def max_freq(self, new_max_freq):
        self._max_freq = new_max_freq

    @min_freq_coord.setter
    def min_freq_coord(self, new_min_freq_coord):
        self._min_freq_coord = new_min_freq_coord

    @max_freq_coord.setter
    def max_freq_coord(self, new_max_freq_coord):
        self._max_freq_coord = new_max_freq_coord

    @avg_freq.setter
    def avg_freq(self, new_avg_freq):
        self._avg_freq = new_avg_freq

    @bandwidth.setter
    def bandwidth(self, new_bandwidth):
        self._bandwidth = new_bandwidth

    @min_intensity.setter
    def min_intensity(self, new_min_intensity):
        self._min_intensity = new_min_intensity

    @max_intensity.setter
    def max_intensity(self, new_max_intensity):
        self._max_intensity = new_max_intensity

    @avg_intensity.setter
    def avg_intensity(self, new_avg_intensity):
        self._avg_intensity = new_avg_intensity

    @bg_intensity.setter
    def bg_intensity(self, new_bg_intensity):
        self._bg_intensity = new_bg_intensity

    @area.setter
    def area(self, new_area):
        self._area = new_area

    @centroid.setter
    def centroid(self, new_centroid):
        self._centroid = new_centroid

    @orientation.setter
    def orientation(self, new_orientation):
        self._orientation = new_orientation

    @coords.setter
    def coords(self, new_coords):
        self._coords = new_coords

    @spectrogram.setter
    def spectrogram(self, new_spectrogram):
        self._spectrogram = new_spectrogram

    @mask.setter
    def mask(self, new_mask):
        self._mask = new_mask

    @cnn_mask.setter
    def cnn_mask(self, new_cnn_mask):
        self._cnn_mask = new_cnn_mask

    @probabilities.setter
    def probabilities(self, new_probabilities):
        self._probabilities = new_probabilities

    @top1.setter
    def top1(self, new_top1):
        self._top1 = new_top1

    @top2.setter
    def top2(self, new_top2):
        self._top2 = new_top2

    def save_spectrogram_as_image(self, path=None, filename="vocal"):
        filename = filename + "_" + str(self.bin_number)
        img = numpy_to_grayscale_image(self.spectrogram)
        save_image_to_disk(image=img, path=join(path, "spectrogram"), filename=filename, img_format="png")

    def save_spectrogram_with_segmentation_as_image(self, path=None, filename="vocal"):
        img = scatter_over_spectrogram(self.spectrogram, self.coords)
        filename = filename + "_" + str(self.bin_number)
        save_image_to_disk(image=img, path=join(path, "spectrogram_validation"), filename=filename, img_format="png")

    def save_mask_as_image(self, path=None, filename="vocal"):
        filename = filename + "_" + str(self.bin_number)
        img = numpy_to_grayscale_image(self.mask)
        save_image_to_disk(image=img, path=join(path, "mask"), filename=filename, img_format="png")

    def save_cnn_mask_as_image(self, path=None, filename="vocal"):
        filename = filename + "_" + str(self.bin_number)
        img = numpy_to_grayscale_image(self.cnn_mask)
        save_image_to_disk(image=img, path=join(path, "cnn_mask"), filename=filename, img_format="png")
