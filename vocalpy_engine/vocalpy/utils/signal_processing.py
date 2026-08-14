# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import numpy as np
from scipy.signal import spectrogram, get_window, windows, butter, lfilter


def compute_spectrogram(
    samples,
    fs,
    window_type="hamming",
    window_size=512,
    noverlap=256,
    nfft=1024,
    lower_frequency_cutoff=None,
    higher_frequency_cutoff=None,
):
    """
    Computes the spectrogram, applies a frequency cutoff, and converts power values to decibel

    Parameters
    ----------
    samples : ndarray
        audio samples time series
    fs : int
        sampling frequency of the audio
    window_type : str (optional)
        windowing function
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window
    window_size : int (optional)
        window size
    noverlap : int (optional)
        number of points to overlap between segments
    nfft : int (optional)
        number of points to compute the stft
    lower_frequency_cutoff : int (optional)
        frequencies lower than this limit will be removed
    higher_frequency_cutoff : int (optional)
        frequencies higher than this limit will be removed

    Returns
    -------
    (f, t, Pxx) : (ndarray, ndarray, ndarray)
        f contains the frequency bins
        t contains the time bins
        Pxx contains the power values for each bin in decibel
    """
    # -- compute spectrogram
    f, t, Pxx = spectrogram(
        x=samples, fs=fs, window=get_window(window_type, window_size), noverlap=noverlap, nfft=nfft, mode="psd"
    )

    # -- apply frequency cutoffs
    if lower_frequency_cutoff is not None:
        Pxx = filter_high_pass(Pxx, f, lower_frequency_cutoff)
        f = filter_high_pass(f, f, lower_frequency_cutoff)
    if higher_frequency_cutoff is not None:
        Pxx = filter_low_pass(Pxx, f, higher_frequency_cutoff)
        f = filter_low_pass(f, f, higher_frequency_cutoff)

    # -- convert to dB
    Pxx = 10 * np.log10(Pxx)

    return f, t, Pxx


def compute_multitaper_spectrogram(
    samples,
    fs,
    window_size=512,
    window_halfbandwidth=4,
    window_count=6,
    noverlap=256,
    nfft=1024,
    lower_frequency_cutoff=None,
    higher_frequency_cutoff=None,
):
    """
    Computes a multitaper spectrogram, applies a frequency cutoff, and converts power values to decibel

    Parameters
    ----------
    samples : ndarray
        audio samples time series
    fs : int
        sampling frequency of the audio
    window_size : int (optional)
        window size in points
    window_halfbandwidth : int (optional)
        time halfbandwith for spheroidal sequences (2*NW = BW/f0)
    window_count : int (optional)
        number of windows (spheroidak sequences) to use
    noverlap : int (optional)
        number of points to overlap between segments
    nfft : int (optional)
        number of points to compute the stft
    lower_frequency_cutoff : int (optional)
        frequencies lower than this limit will be removed
    higher_frequency_cutoff : int (optional)
        frequencies higher than this limit will be removed

    Returns
    -------
    (f, t, Pxx) : (ndarray, ndarray, ndarray)
        f contains the frequency bins
        t contains the time bins
        Pxx contains the power values for each bin in decibel
    """
    # -- get dpss windows for multitaper spectrogram
    wins = windows.dpss(window_size, window_halfbandwidth, window_count)

    # -- compute spectrogram using each window and average them
    for idx, window in enumerate(wins):
        # -- compute spectrogram using each window
        f, t, Pxx = spectrogram(x=samples, fs=fs, window=window, noverlap=noverlap, nfft=nfft, mode="psd")

        # -- apply frequency cutoffs
        if lower_frequency_cutoff is not None:
            Pxx = filter_high_pass(Pxx, f, lower_frequency_cutoff)
            f = filter_high_pass(f, f, lower_frequency_cutoff)
        if higher_frequency_cutoff is not None:
            Pxx = filter_low_pass(Pxx, f, higher_frequency_cutoff)
            f = filter_low_pass(f, f, higher_frequency_cutoff)

        if idx == 0:
            Pxx_average = 10 * np.log10(Pxx) / window_count
        else:
            Pxx_average += 10 * np.log10(Pxx) / window_count

    return f, t, Pxx_average


def filter_low_pass(data, frequencies, frequency_cutoff):
    return data[(frequencies < frequency_cutoff)]


def filter_high_pass(data, frequencies, frequency_cutoff):
    return data[(frequencies > frequency_cutoff)]


def filter_band_pass(data, frequencies, lower_frequency_cutoff, higher_frequency_cutoff):
    return data[(frequencies > lower_frequency_cutoff) & (frequencies < higher_frequency_cutoff)]


def butter_bandpass(lower_frequency_cutoff, higher_frequency_cutoff, fs, order=25):
    """
    Creates a Butterworth bandpass filter

    Parameters
    ----------
    lower_frequency_cutoff : int
        lower frequency cutoff for the butterworth filter
    higher_frequency_cutoff
        higher frequency cutoff for the butterworth filter
    fs : int
        signal sampling rate
    order : int, optional
        Butterworth filter order

    Returns
    -------
    (b, a) : (ndarray, ndarray)
        numerator (b) and denominator (a) polynomials of the IIR filter
    """
    nyquist_freq = 0.5 * fs
    low = lower_frequency_cutoff / nyquist_freq
    high = higher_frequency_cutoff / nyquist_freq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(samples, lower_frequency_cutoff, higher_frequency_cutoff, fs, order=25):
    """
    Applies a Butterworth bandpass filter to a signal

    Parameters
    ----------
    samples : ndarray
        source signal to be filtered
    lower_frequency_cutoff : int
        lower frequency cutoff for the butterworth filter
    higher_frequency_cutoff
        higher frequency cutoff for the butterworth filter
    fs : int
        signal sampling rate
    order : int, optional
        Butterworth filter order

    Returns
    -------
    y : ndarray
        source signal filtered using the butterworth filter
    """
    b, a = butter_bandpass(lower_frequency_cutoff, higher_frequency_cutoff, fs, order=order)
    y = lfilter(b, a, samples)
    return y


def butter_highpass(higher_frequency_cutoff, fs, order=25):
    """
    Creates a Butterworth highpass filter

    Parameters
    ----------
    higher_frequency_cutoff
        higher frequency cutoff for the butterworth filter
    fs : int
        signal sampling rate
    order : int, optional
        Butterworth filter order

    Returns
    -------
    (b, a) : (ndarray, ndarray)
        numerator (b) and denominator (a) polynomials of the IIR filter
    """
    nyquist_freq = 0.5 * fs
    high = higher_frequency_cutoff / nyquist_freq
    b, a = butter(order, high, btype="high")
    return b, a


def butter_highpass_filter(samples, higher_frequency_cutoff, fs, order=25):
    """
    Applies a Butterworth highpass filter to a signal

    Parameters
    ----------
    samples : ndarray
        source signal to be filtered
    higher_frequency_cutoff
        higher frequency cutoff for the butterworth filter
    fs : int
        signal sampling rate
    order : int, optional
        Butterworth filter order

    Returns
    -------
    y : ndarray
        source signal filtered using the butterworth filter
    """
    b, a = butter_highpass(higher_frequency_cutoff, fs, order=order)
    y = lfilter(b, a, samples)
    return y


def butter_lowpass(lower_frequency_cutoff, fs, order=25):
    """
    Creates a Butterworth lowpass filter

    Parameters
    ----------
    lower_frequency_cutoff : int
        lower frequency cutoff for the butterworth filter
    fs : int
        signal sampling rate
    order : int, optional
        Butterworth filter order

    Returns
    -------
    (b, a) : (ndarray, ndarray)
        numerator (b) and denominator (a) polynomials of the IIR filter
    """
    nyquist_freq = 0.5 * fs
    high = lower_frequency_cutoff / nyquist_freq
    b, a = butter(order, high, btype="low")
    return b, a


def butter_lowpass_filter(samples, lower_frequency_cutoff, fs, order=25):
    """
    Applies a Butterworth lowpass filter to a signal

    Parameters
    ----------
    samples : ndarray
        source signal to be filtered
    lower_frequency_cutoff : int
        lower frequency cutoff for the butterworth filter
    fs : int
        signal sampling rate
    order : int, optional
        Butterworth filter order

    Returns
    -------
    y : ndarray
        source signal filtered using the butterworth filter
    """
    b, a = butter_highpass(lower_frequency_cutoff, fs, order=order)
    y = lfilter(b, a, samples)
    return y
