# -*- coding: utf-8 -*-
"""Metadata and validation helpers for bundled pretrained models."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import hashlib
import json

from dataclasses import dataclass
from pathlib import Path


PRETRAINED_DIR = Path(__file__).with_name("pretrained")
PRETRAINED_MODEL_METADATA_FILES = sorted(PRETRAINED_DIR.glob("*.metadata.json"))


@dataclass(frozen=True)
class PretrainedModelSpec:
    network_type: str
    filename: str
    architecture: str
    num_classes: int
    classes: tuple[str, ...]
    input_shape: tuple[int, ...]
    checkpoint_keys: tuple[str, ...]
    prediction_type: str | None
    default_threshold: float | None
    sha256: str
    storage: str
    source: str
    source_version: str
    repo_history: dict[str, str]
    notes: str

    @property
    def path(self) -> Path:
        return PRETRAINED_DIR / self.filename

    @property
    def metadata_path(self) -> Path:
        checkpoint_name = self.filename
        if checkpoint_name.endswith(".pth.tar"):
            checkpoint_name = checkpoint_name[: -len(".pth.tar")]
        else:
            checkpoint_name = Path(checkpoint_name).stem
        return PRETRAINED_DIR / f"{checkpoint_name}.metadata.json"


def _load_pretrained_model_spec(path: str | Path) -> PretrainedModelSpec:
    metadata = json.loads(Path(path).read_text())
    return PretrainedModelSpec(
        network_type=metadata["network_type"],
        filename=metadata["filename"],
        architecture=metadata["architecture"],
        num_classes=int(metadata["num_classes"]),
        classes=tuple(metadata["classes"]),
        input_shape=tuple(metadata["input_shape"]),
        checkpoint_keys=tuple(metadata["checkpoint_keys"]),
        prediction_type=metadata.get("prediction_type"),
        default_threshold=(
            float(metadata["default_threshold"]) if metadata.get("default_threshold") is not None else None
        ),
        sha256=metadata["sha256"],
        storage=metadata["storage"],
        source=metadata["source"],
        source_version=metadata["source_version"],
        repo_history=metadata.get("repo_history", {}),
        notes=metadata["notes"],
    )


def load_pretrained_model_specs() -> dict[str, PretrainedModelSpec]:
    specs = {}
    for metadata_path in PRETRAINED_MODEL_METADATA_FILES:
        spec = _load_pretrained_model_spec(metadata_path)
        specs[spec.network_type] = spec
    return specs


PRETRAINED_MODEL_SPECS = load_pretrained_model_specs()


def get_pretrained_model_spec(network_type: str) -> PretrainedModelSpec:
    try:
        return PRETRAINED_MODEL_SPECS[network_type]
    except KeyError as exc:
        raise ValueError(f"unsupported network_type: {network_type}") from exc


def compute_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pretrained_model_file(path: str | Path, expected_sha256: str | None = None) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"pretrained model does not exist: {path}")

    actual_sha256 = compute_sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"pretrained model checksum mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return actual_sha256
