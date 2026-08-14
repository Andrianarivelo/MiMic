# -*- coding: utf-8 -*-
"""Behavior tests for I/O helpers."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import pickle

import pytest
import torch

from vocalpy.errors import InputPathError, SerializationError
from vocalpy.modules.list_of_vocals import ListOfVocals
from vocalpy.modules.recording import Recording
from vocalpy.modules.vocal import Vocal
from vocalpy.utils.io import (
    VOCALPY_SERIALIZATION_FORMAT,
    VOCALPY_SERIALIZATION_VERSION,
    create_directory,
    create_output_directory_structure,
    get_output_directory_for_audio_file,
    load_checkpoint,
    load_model,
    load_pickle_file,
    load_recording_data,
    load_vocalpy_file,
    parse_input_path,
    rewrite_vocalpy_file,
    remove_directory,
    write_pickle_file,
)


def test_write_and_load_pickle_file_round_trip(tmp_path):
    payload = {"detected_vocals": 3}

    file_path = write_pickle_file(payload, "recording", tmp_path, object_type="recording")
    loaded = load_pickle_file("recording", tmp_path)

    with file_path.open("rb") as input_file:
        envelope = pickle.load(input_file)

    assert envelope["format"] == VOCALPY_SERIALIZATION_FORMAT
    assert envelope["format_version"] == VOCALPY_SERIALIZATION_VERSION
    assert envelope["object_type"] == "recording"
    assert loaded == payload


def test_load_recording_data_round_trip(tmp_path):
    path = write_pickle_file({"name": "mouse"}, "recording_without_spectrograms", tmp_path, object_type="recording")

    loaded = load_recording_data(path)

    assert loaded == {"name": "mouse"}


def test_load_recording_data_round_trip_strips_transient_visual_payloads(tmp_path):
    vocal = Vocal(
        bin_number=1,
        spectrogram=torch.ones((2, 2), dtype=torch.uint8).numpy(),
        mask=torch.ones((2, 2), dtype=torch.uint8).numpy(),
        cnn_mask=torch.ones((2, 2), dtype=torch.uint8).numpy(),
    )
    recording = Recording.__new__(Recording)
    recording._list_of_vocals = ListOfVocals([vocal])
    recording._has_list_of_vocals = True

    recording.remove_spectrograms_and_masks_from_object()
    recording.save_recording_object(tmp_path, filename="recording_without_spectrograms")

    loaded = load_recording_data(tmp_path / "recording_without_spectrograms.vocalpy")
    loaded_vocal = loaded.list_of_vocals.vocals_in_recording[0]

    assert loaded_vocal.spectrogram is None
    assert loaded_vocal.mask is None
    assert loaded_vocal.cnn_mask is None


def test_load_vocalpy_file_keeps_legacy_pickle_compatibility(tmp_path):
    path = tmp_path / "legacy.vocalpy"
    payload = {"legacy": True}
    with path.open("wb") as output_file:
        pickle.dump(payload, output_file)

    loaded = load_vocalpy_file(path, expected_object_type="recording")

    assert loaded == payload


def test_load_vocalpy_file_rejects_type_mismatch(tmp_path):
    path = write_pickle_file({"name": "mouse"}, "recording_without_spectrograms", tmp_path, object_type="recording")

    with pytest.raises(SerializationError, match="type mismatch"):
        load_vocalpy_file(path, expected_object_type="list_of_vocals")


def test_load_vocalpy_file_rejects_unknown_format_version(tmp_path):
    path = tmp_path / "broken.vocalpy"
    payload = {
        "format": VOCALPY_SERIALIZATION_FORMAT,
        "format_version": VOCALPY_SERIALIZATION_VERSION + 1,
        "object_type": "recording",
        "payload": {"broken": True},
    }
    with path.open("wb") as output_file:
        pickle.dump(payload, output_file)

    with pytest.raises(SerializationError, match="unsupported VocalPy serialization version"):
        load_vocalpy_file(path, expected_object_type="recording")


def test_rewrite_vocalpy_file_upgrades_legacy_pickle_to_current_envelope(tmp_path):
    path = tmp_path / "legacy.vocalpy"
    payload = {"legacy": True}
    with path.open("wb") as output_file:
        pickle.dump(payload, output_file)

    rewrite_vocalpy_file(path, expected_object_type="recording")

    with path.open("rb") as input_file:
        envelope = pickle.load(input_file)

    assert envelope["format"] == VOCALPY_SERIALIZATION_FORMAT
    assert envelope["object_type"] == "recording"
    assert load_vocalpy_file(path, expected_object_type="recording") == payload


def test_load_checkpoint(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with torch.no_grad():
        model.weight.fill_(0.5)
        model.bias.fill_(0.25)

    checkpoint = {
        "state_dict": model.state_dict(),
        "optim_dict": optimizer.state_dict(),
    }

    loaded_model = torch.nn.Linear(2, 1)
    loaded_optimizer = torch.optim.SGD(loaded_model.parameters(), lr=0.1)

    checkpoint_path = tmp_path / "checkpoint.pth.tar"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_data = load_checkpoint(checkpoint_path, loaded_model, "cpu", loaded_optimizer)

    assert "state_dict" in checkpoint_data
    assert torch.equal(loaded_model.weight, model.weight)
    assert torch.equal(loaded_model.bias, model.bias)


def test_load_checkpoint_raises_for_missing_file():
    model = torch.nn.Linear(2, 1)

    with pytest.raises(FileNotFoundError):
        load_checkpoint("/tmp/does-not-exist-checkpoint.pth.tar", model, "cpu")


def test_load_checkpoint_raises_for_missing_state_dict(tmp_path):
    checkpoint_path = tmp_path / "broken-checkpoint.pth.tar"
    torch.save({"weights": {}}, checkpoint_path)
    model = torch.nn.Linear(2, 1)

    with pytest.raises(ValueError, match="state_dict"):
        load_checkpoint(checkpoint_path, model, "cpu")


def test_load_model_round_trip(tmp_path):
    checkpoint_path = tmp_path / "model.pth.tar"
    payload = {"weights": torch.tensor([1.0, 2.0])}
    torch.save(payload, checkpoint_path)

    loaded = load_model(checkpoint_path, "cpu")

    assert torch.equal(loaded["weights"], payload["weights"])


def test_parse_input_path_accepts_single_audio_file(tmp_path):
    audio_path = tmp_path / "mouse.wav"
    audio_path.write_bytes(b"audio")

    files = parse_input_path(audio_path)

    assert files == [str(audio_path)]


def test_parse_input_path_collects_supported_audio_extensions_recursively(tmp_path):
    (tmp_path / "nested").mkdir()
    wav_path = tmp_path / "nested" / "mouse.wav"
    flac_path = tmp_path / "rat.FLAC"
    wav_path.write_bytes(b"wav")
    flac_path.write_bytes(b"flac")

    files = parse_input_path(tmp_path, search_tree=True)

    assert files == sorted([str(wav_path), str(flac_path)])


def test_parse_input_path_raises_for_missing_path():
    with pytest.raises(InputPathError, match="usage: vocalpy"):
        parse_input_path()

    with pytest.raises(InputPathError, match="not a file or directory"):
        parse_input_path("/tmp/does-not-exist")


def test_create_output_directory_structure_uses_shared_output_path_helper():
    files = ["/tmp/mouse.wav", "/tmp/nested/rat.FLAC"]

    output_dirs = create_output_directory_structure(files)

    assert output_dirs == ["/tmp/mouse_outputs", "/tmp/nested/rat_outputs"]
    assert get_output_directory_for_audio_file("/tmp/mouse.wav") == "/tmp/mouse_outputs"


def test_create_directory_is_idempotent(tmp_path):
    path = tmp_path / "outputs" / "spectrogram"

    create_directory(path)
    create_directory(path)

    assert path.is_dir()


def test_remove_directory_removes_nested_tree(tmp_path):
    path = tmp_path / "outputs" / "spectrogram"
    path.mkdir(parents=True)
    (path / "frame.png").write_bytes(b"data")

    remove_directory(path.parent)

    assert not path.parent.exists()
