# -*- coding: utf-8 -*-
"""Command-line entry point for VocalPy."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

from time import time
from argparse import ArgumentParser

from vocalpy.errors import VocalPyError
from vocalpy.modules.recording import Recording
from vocalpy.utils.misc import create_logger, validate_arguments
from vocalpy.utils.io import parse_input_path, create_output_directory_structure, create_directory


def int_or_default(value):
    if value == "default":
        return value
    return int(value)


def float_or_default(value):
    if value == "default":
        return value
    return float(value)


def build_parser():
    p = ArgumentParser()
    p.add_argument("-a", "--animal", help="choose from ['mouse', 'rat']", type=str, default="mouse")
    p.add_argument("-p", "--path_to_audio", help="path to audio file or directory", type=str, default=None)
    p.add_argument(
        "-b",
        "--bin_size",
        help="split audio into segments for parallel processing (in seconds; default=60)",
        type=int,
        default=60,
    )
    p.add_argument(
        "-lf", "--lower_frequency_cutoff", help="lower frequency cutoff", type=int_or_default, default="default"
    )
    p.add_argument(
        "-hf", "--higher_frequency_cutoff", help="higher frequency cutoff", type=int_or_default, default="default"
    )
    p.add_argument("-t", "--threads", help="number of threads to use (default=max/2)", type=int, default=-1)
    p.add_argument("-v", "--verbose", help="enable verbose output", action="store_true")
    p.add_argument("-l", "--validation", help="saves overlay of segmentation for manual verifcation", action="store_true")
    p.add_argument("--segmenter", help="enable SqueakOut neural segmentation over detected vocal crops", action="store_true")
    p.add_argument(
        "--segmentation_model_path",
        help="path to a SqueakOut checkpoint; defaults to the bundled checkpoint",
        type=str,
        default=None,
    )
    p.add_argument(
        "--segmentation_threshold",
        help="probability threshold for binary vocal masks",
        type=float_or_default,
        default="default",
    )
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args = validate_arguments(args)
        list_of_files = parse_input_path(args.path_to_audio)
    except VocalPyError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")

    list_of_output_dirs = create_output_directory_structure(list_of_files)

    time_all_recordings = time()
    logger = None
    for file_idx, __ in enumerate(list_of_files):
        audio_path = list_of_files[file_idx]
        output_dir = list_of_output_dirs[file_idx]

        create_directory(output_dir)
        logger = create_logger(args, output_dir)
        logger.info(f"selected file:\n{audio_path}")
        logger.info(f"output files will be saved to:\n{output_dir}")
        logger.info(f"selected animal pipeline: {args.animal}")

        time_start = time()
        recording = Recording(recording_path=audio_path, args=args)
        logger.info(f"recording object created ({time() - time_start:.2f}s)")
        logger.info(recording)
        logger.info(f"recording duration: {recording.audio.audio_duration:.2f} seconds")
        logger.info(f"splitting audio into {recording.audio.bins} chunks")

        recording.identify_vocalizations()
        recording.classify_vocalizations()
        recording.segment_vocalizations()
        recording.save_outputs(validation_flag=args.validation)

        logger.info(f"total runtime: {(time() - time_start) // 60:.0f}m {(time() - time_start) % 60:.0f}s")

    if logger is not None:
        logger.info(
            "total runtime for all recordings: "
            f"{(time() - time_all_recordings) // 60:.0f}m {(time() - time_all_recordings) % 60:.0f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
