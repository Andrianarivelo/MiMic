# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import glob
import yaml
import shutil
import pickle
import tempfile

import numpy as np
import soundfile as sf

from os import makedirs
from os.path import basename, exists, isdir, isfile, join, splitext
from pathlib import Path

from vocalpy import __version__
from vocalpy.errors import InputPathError, SerializationError


VOCALPY_FILE_EXTENSION = ".vocalpy"
VOCALPY_SERIALIZATION_FORMAT = "vocalpy-object"
VOCALPY_SERIALIZATION_VERSION = 1


def get_vocalpy_file_path(filename, path):
    return Path(path) / f"{filename}{VOCALPY_FILE_EXTENSION}"


def _infer_serialized_object_type(payload):
    class_name = payload.__class__.__name__.lower()
    if class_name == "recording":
        return "recording"
    if class_name == "listofvocals":
        return "list_of_vocals"
    return class_name


def _build_serialization_envelope(payload, object_type):
    return {
        "format": VOCALPY_SERIALIZATION_FORMAT,
        "format_version": VOCALPY_SERIALIZATION_VERSION,
        "package_version": __version__,
        "object_type": object_type,
        "payload": payload,
    }


def _is_serialization_envelope(value):
    return isinstance(value, dict) and value.get("format") == VOCALPY_SERIALIZATION_FORMAT


def _unwrap_serialized_payload(value, source, expected_object_type=None):
    if not _is_serialization_envelope(value):
        return value

    format_version = value.get("format_version")
    if format_version != VOCALPY_SERIALIZATION_VERSION:
        raise SerializationError(
            f"unsupported VocalPy serialization version {format_version} in {source}; "
            f"expected {VOCALPY_SERIALIZATION_VERSION}"
        )

    object_type = value.get("object_type")
    if expected_object_type is not None and object_type != expected_object_type:
        raise SerializationError(
            f"serialized object type mismatch in {source}: "
            f"expected {expected_object_type}, found {object_type}"
        )

    if "payload" not in value:
        raise SerializationError(f"serialized object payload missing in {source}")

    return value["payload"]


def write_vocalpy_file(payload, filename, path, object_type=None):
    """
    Serialize a VocalPy object with versioned metadata.
    """
    if exists(path) is False:
        raise ValueError(f"path does not existe: {path}")

    resolved_object_type = object_type or _infer_serialized_object_type(payload)
    file_path = get_vocalpy_file_path(filename, path)
    envelope = _build_serialization_envelope(payload, resolved_object_type)
    with file_path.open("wb") as output_file:
        pickle.dump(envelope, output_file)
    return file_path


def load_vocalpy_file(path, expected_object_type=None):
    """
    Load a VocalPy serialized object with legacy raw-pickle compatibility.
    """
    file_path = Path(path)
    if file_path.exists() is False:
        raise ValueError(f"file does not existe: {path}")

    with file_path.open("rb") as input_file:
        value = pickle.load(input_file)
    return _unwrap_serialized_payload(value, file_path, expected_object_type=expected_object_type)


def rewrite_vocalpy_file(path, expected_object_type=None, object_type=None):
    """
    Rewrite a VocalPy artifact through the current versioned envelope.
    """
    file_path = Path(path)
    payload = load_vocalpy_file(file_path, expected_object_type=expected_object_type)
    resolved_object_type = object_type or expected_object_type or _infer_serialized_object_type(payload)
    envelope = _build_serialization_envelope(payload, resolved_object_type)

    with tempfile.NamedTemporaryFile("wb", dir=file_path.parent, delete=False) as temp_file:
        pickle.dump(envelope, temp_file)
        temp_path = Path(temp_file.name)
    temp_path.replace(file_path)
    return file_path


def write_pickle_file(file, filename, path, object_type=None):
    """
    Writes vocalpy pickle object to a path

    Parameters
    ----------
    file : Object
        object to be saved
    filename : str
        filename to be used
    path : str
        path to save the object

    Raises
    ------
    ValueError
        if the path does not exist
    """
    return write_vocalpy_file(file, filename, path, object_type=object_type)


def load_pickle_file(filename, path, expected_object_type=None):
    """
    Loads vocalpy pickle object from a path

    Parameters
    ----------
    filename : str
        object filename
    path : str
        path to object

    Raises
    ------
    ValueError
        if the path does not exist
    """
    if exists(path) is False:
        raise ValueError(f"path does not existe: {path}")

    return load_vocalpy_file(
        get_vocalpy_file_path(filename, path),
        expected_object_type=expected_object_type,
    )


def load_recording_data(path):
    """
    Loads vocalpy recording object from a path

    Parameters
    ----------
    path : str
        full path to object

    Raises
    ------
    ValueError
        if the file does not exist
    """
    return load_vocalpy_file(path, expected_object_type="recording")


def load_checkpoint(checkpoint, model, device, optimizer=None):
    """Loads model parameters (state_dict) from file_path.
    If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    import torch

    if not exists(checkpoint):
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

    checkpoint_data = torch.load(checkpoint, map_location=torch.device(device))
    if not isinstance(checkpoint_data, dict) or "state_dict" not in checkpoint_data:
        raise ValueError(f"checkpoint is missing state_dict: {checkpoint}")

    model.load_state_dict(checkpoint_data["state_dict"])

    if optimizer is not None:
        if "optim_dict" not in checkpoint_data:
            raise ValueError(f"checkpoint is missing optim_dict: {checkpoint}")
        optimizer.load_state_dict(checkpoint_data["optim_dict"])

    return checkpoint_data


def load_model(model_path, device):
    """
    Directly load a pretrained pytorch model

    Parameters
    ----------
    model_path : str
        path to model
    device : torch.device
        device to run (CPU or GPU)
    """
    import torch

    try:
        return torch.load(model_path, map_location=torch.device(device), weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=torch.device(device))


def parse_input_path(path=None, search_tree=False):
    """
    Parse input path. If it is a directory, return list of files; if it
    is a file, return the file path

    Parameters
    ----------
    path : str
        path provided by the user
    """
    if path is None:
        raise InputPathError("usage: vocalpy --path_to_audio='/path/to/audio'")
    if isdir(path):
        if search_tree:
            types = (
                join(path, "**/*.wav"),
                join(path, "**/*.WAV"),
                join(path, "**/*.flac"),
                join(path, "**/*.FLAC"),
            )
        else:
            types = (
                join(path, "*.wav"),
                join(path, "*.WAV"),
                join(path, "*.flac"),
                join(path, "*.FLAC"),
            )
        files_found = []
        for files in types:
            files_found.extend(glob.glob(files, recursive=search_tree))
        return sorted(set(files_found))
    if isfile(path):
        return [str(path)]

    raise InputPathError(f"audio path is not a file or directory: {path}")


def get_output_directory_for_audio_file(path):
    basepath, _ = splitext(path)
    return basepath + "_outputs"


def create_output_directory_structure(list_of_files):
    """
    Creates directory structure for output files from VocalPy

    Parameters
    ----------
    list_of_files : List[str]
        list of files provided by the user
    """

    return [get_output_directory_for_audio_file(file) for file in list_of_files]


def create_directory(path):
    """
    Creates a directory at the provided path

    Parameters
    ----------
    path : str
        path to be created
    """
    if not exists(path):
        makedirs(path, exist_ok=True)
    return 0


def remove_directory(path):
    """
    Removes a directory at the provided path

    Parameters
    ----------
    path : str
        path to be removed
    """
    shutil.rmtree(path, ignore_errors=True)
    return 0


def read_yaml(path_to_file):
    """
    Loads a YAML configuration file

    Parameters
    ----------
    path_to_file : str
        path to YAML file

    Returns
    -------
    yml_data : dict
        YAML file data read into a dictionary
    """
    with open(path_to_file, "r") as ymlfile:
        yml_data = yaml.safe_load(ymlfile)
    return yml_data


def write_yaml(data, path_to_file):
    """
    Writes a YAML configuration file

    Parameters
    ----------
    data : dict
        dict data to write as a YAML file
    path_to_file : str
        path to YAML file

    """
    with open(path_to_file, "w") as ymlfile:
        yaml.dump(data, ymlfile, sort_keys=False)


def read_audio(path_to_file, start=0, stop=None):
    """
    Reads audio and metadata using SoundFile

    Parameters
    ----------
    path_to_file : str
        path to audio file

    Returns
    -------
    (samples, sample_rate) : (ndarray, int)
        mono audio samples (always first channel) and audio sampling frequency
    """
    samples, sample_rate = sf.read(path_to_file, start=start, stop=stop, always_2d=True)
    return samples[:, 0], sample_rate


def read_audio_information(path_to_file):
    """
    Reads audio metadata using SoundFile

    Parameters
    ----------
    path_to_file : str
        path to audio file

    Returns
    -------
    metadata : dict
        returns audio metadata in a dictionary including:
            path to file
            sampling rate
            number of channels
            duration
            format
    """
    return sf.info(path_to_file)


def save_image_to_disk(image, path, filename, img_format="png"):
    """
    Saves PIL Image to disk

    Parameters
    ----------
    image : :class:`PIL.Image`
        image to be saved to disk
    path : str
        target path
    filename : str
        image file name
    img_format : str, optional
        image encoding format (png, jpg, gif, ...)

    Raises
    ------
    ValueError
        if target path does not exist
    """
    if exists(path) is False:
        raise ValueError(f"path does not exist: {path}")
    image.save(join(path, filename + "." + img_format))


def save_dataframe_as_csv(dataframe, path, filename):
    """
    Saves a Pandas DataFrame to disk

    Parameters
    ----------
    dataframe : :class:`pandas.DataFrame`
        dataframe to be saved to disk
    path : str
        target path
    filename : str
        image file name

    Raises
    ------
    ValueError
        if target path does not exist
    """
    if exists(path) is False:
        raise ValueError(f"path does not exist: {path}")
    # -- start index from 1 instead of 0
    dataframe.index = np.arange(1, len(dataframe) + 1)
    dataframe.to_csv(
        join(path, splitext(filename)[0] + "_stats.csv"), float_format="%.6f",
    )
