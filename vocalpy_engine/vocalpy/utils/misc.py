# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import logging
import warnings

import pandas as pd

from os.path import join
from multiprocessing import cpu_count

from vocalpy.errors import ValidationError
from vocalpy.utils.io import read_yaml


SUPPORTED_ANIMALS = ("mouse", "rat", "guineapig")
DEFAULT_FREQUENCY_CUTOFFS = {
    "mouse": (45000, 125000),
    "rat": (18000, 125000),
    "guineapig": (0, 22000),
}


def _get_frequency_arg_names(args):
    if hasattr(args, "lower_frequency_cutoff") and hasattr(args, "higher_frequency_cutoff"):
        return "lower_frequency_cutoff", "higher_frequency_cutoff"
    if hasattr(args, "lower_frequency") and hasattr(args, "higher_frequency"):
        return "lower_frequency", "higher_frequency"
    raise AttributeError("args must expose lower/higher frequency cutoff fields")


def _get_frequency_range_args(args):
    lower_name, higher_name = _get_frequency_arg_names(args)
    return getattr(args, lower_name), getattr(args, higher_name)


def _set_frequency_range_args(args, lower_frequency, higher_frequency):
    lower_name, higher_name = _get_frequency_arg_names(args)
    setattr(args, lower_name, lower_frequency)
    setattr(args, higher_name, higher_frequency)


def create_logger(args=None, out_dir=None):
    """
    Creates a logger to log information during execution
    """
    log_path = join(out_dir, "output.log")
    handlers = [logging.FileHandler(f"{out_dir}/output.log")]
    if args.verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logger = logging.getLogger()
    logger.info(f"logging to file: {log_path}")
    if args.verbose:
        logger.info("verbose output on")
    return logger


def validate_arguments(args):
    """
    Validates arguments passed by the user

    Parameter
    ---------
    args : ArgumentParser
    """
    validate_animal(args.animal)
    validate_bin_size(args.bin_size)
    lower_frequency, higher_frequency = _get_frequency_range_args(args)
    validated_lower, validated_higher = validate_frequency_range(
        lower_frequency, higher_frequency, args.animal
    )
    _set_frequency_range_args(args, validated_lower, validated_higher)
    validate_thread_count(args.threads)
    validate_segmentation_arguments(args)
    return args


def validate_animal(animal):
    if animal not in SUPPORTED_ANIMALS:
        supported = ", ".join(SUPPORTED_ANIMALS)
        raise ValidationError(f"unsupported animal pipeline '{animal}'. available pipelines: {supported}")
    return 0


def validate_bin_size(bin_size):
    if bin_size <= 0:
        raise ValidationError(f"bin_size must be a positive integer. provided value: {bin_size}")
    return 0


def validate_frequency_range(lower_frequency, higher_frequency, animal):
    default_low, default_high = DEFAULT_FREQUENCY_CUTOFFS[animal]
    low_freq = default_low if lower_frequency == "default" else int(lower_frequency)
    high_freq = default_high if higher_frequency == "default" else int(higher_frequency)
    if low_freq < 0:
        raise ValidationError(f"lower frequency cutoff must be non-negative. provided value: {low_freq}")
    if high_freq != -1 and high_freq < 0:
        raise ValidationError(
            f"higher frequency cutoff must be -1 or a non-negative integer. provided value: {high_freq}"
        )
    if low_freq > high_freq and high_freq != -1:
        raise ValidationError(
            "low frequency cutoff must be lower than the high frequency cutoff. "
            f"provided values: low_freq={low_freq}; high_freq={high_freq}"
        )
    return low_freq, high_freq


def validate_thread_count(threads):
    num_cores = cpu_count()
    if (threads < -1) or (threads == 0):
        raise ValidationError(
            "number of threads must be a positive integer or -1. "
            f"provided value: {threads}; computer core count: {num_cores}"
        )
    if threads > num_cores:
        warnings.warn(
            "number of threads is equal or higher than number of available cores. "
            "if your CPU has hyperthreading, use number of physical cores for better performance. "
            f"provided value: {threads}; computer thread count: {num_cores}",
            UserWarning,
            stacklevel=2,
        )
    return 0


def validate_segmentation_arguments(args):
    if not hasattr(args, "segmenter"):
        return 0

    threshold = getattr(args, "segmentation_threshold", "default")
    if threshold != "default" and not (0 < float(threshold) < 1):
        raise ValidationError(
            "segmentation threshold must be between 0 and 1. "
            f"provided value: {threshold}"
        )
    return 0


def create_dataframe_from_list_of_vocals(list_of_vocals):
    """
    Creates a Pandas DataFrame from a :class:`ListOfVocals`

    Parameters
    ----------
    list_of_vocals : :class:`ListOfVocals`
        list of vocals is a :class:`ListOfVocals` instance

    Returns
    -------
    dataframe : :class:`pandas.DataFrame`
        dataframe containing all vocals from the list of vocals
    """
    columns = [
        "bin_number",
        "start(s)",
        "end(s)",
        "duration(ms)",
        "interval(s)",
        "min_freq",
        "max_freq",
        "avg_freq",
        "bandwidth",
        "min_intensity",
        "max_intensity",
        "avg_intensity",
        "bg_intensity",
        "area(pixels)",
        "centroid_y",
        "class_top1",
        "class_top2",
    ]
    records = []
    for vocal in list_of_vocals.vocals_in_recording:
        records.append(
            {
                "bin_number": vocal.bin_number,
                "start(s)": vocal.start,
                "end(s)": vocal.end,
                "duration(ms)": vocal.duration,
                "interval(s)": vocal.interval,
                "min_freq": vocal.min_freq,
                "max_freq": vocal.max_freq,
                "avg_freq": vocal.avg_freq,
                "bandwidth": vocal.bandwidth,
                "min_intensity": vocal.min_intensity,
                "max_intensity": vocal.max_intensity,
                "avg_intensity": vocal.avg_intensity,
                "bg_intensity": vocal.bg_intensity,
                "area(pixels)": vocal.area,
                "centroid_y": vocal.centroid[0],
                "class_top1": vocal.top1,
                "class_top2": vocal.top2,
            }
        )

    dataframe = pd.DataFrame.from_records(records, columns=columns)

    # -- sort vocalizations by start time and save csv
    dataframe.sort_values(
        by="start(s)", ascending=True, inplace=True, kind="quicksort", na_position="last",
    )

    return dataframe
