"""Adapter over the vendored ``gumadeiras/vocalpy`` USV detection pipeline.

The GUI and CLI drive the exact same code path the upstream ``vocalpy`` command
uses: build an ``args`` namespace, create a :class:`~vocalpy.modules.recording.Recording`,
run ``identify_vocalizations`` -> ``classify_vocalizations`` -> ``segment_vocalizations``
-> ``save_outputs`` (which writes the per-file ``*_stats.csv``), then read the detected
vocalizations back for display.

This module owns *no* Qt so it can be unit-tested and reused by the CLI.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from argparse import Namespace

import numpy as np

# Vendored engine (installed editable as `vocalpy`).
from vocalpy.modules.recording import Recording
from vocalpy.utils.misc import (
    DEFAULT_FREQUENCY_CUTOFFS,
    SUPPORTED_ANIMALS,
    validate_arguments,
)
from vocalpy.utils.io import get_output_directory_for_audio_file

SPECIES = tuple(SUPPORTED_ANIMALS)                     # ('mouse', 'rat', 'guineapig')
DEFAULT_CUTOFFS = dict(DEFAULT_FREQUENCY_CUTOFFS)

# The 11 mouse USV syllable classes (+ the binary noise/vocal labels) get stable,
# perceptually distinct colors so detection overlays and the table read as one system.
CLASS_COLORS = {
    "chevron": "#38bdf8", "complex": "#a78bfa", "down_fm": "#f472b6",
    "flat": "#34d399", "mult_steps": "#fbbf24", "rev_chevron": "#22d3ee",
    "short": "#fb7185", "step_down": "#818cf8", "step_up": "#2dd4bf",
    "two_steps": "#f59e0b", "up_fm": "#4ade80", "noise": "#64748b",
    "vocal": "#19c8d6", None: "#19c8d6", "": "#19c8d6",
}


def class_color(label: str | None) -> str:
    return CLASS_COLORS.get(label, "#19c8d6")


@dataclasses.dataclass
class DetectParams:
    """User-facing detection settings mirrored onto the engine ``args``."""

    animal: str = "mouse"
    bin_size: int = 60
    lower_frequency_cutoff: object = "default"   # int or "default"
    higher_frequency_cutoff: object = "default"
    threads: int = -1
    segmenter: bool = False
    validation: bool = False
    segmentation_model_path: str | None = None
    segmentation_threshold: object = "default"

    def to_args(self, path: str | pathlib.Path) -> Namespace:
        return Namespace(
            animal=self.animal,
            path_to_audio=str(path),
            bin_size=int(self.bin_size),
            lower_frequency_cutoff=self.lower_frequency_cutoff,
            higher_frequency_cutoff=self.higher_frequency_cutoff,
            threads=int(self.threads),
            verbose=False,
            validation=bool(self.validation),
            segmenter=bool(self.segmenter),
            segmentation_model_path=self.segmentation_model_path,
            segmentation_threshold=self.segmentation_threshold,
        )


@dataclasses.dataclass
class Detection:
    """One detected vocalization, flattened for the table / overlay / metrics."""

    index: int
    bin_number: int
    start: float          # s
    end: float            # s
    duration_ms: float
    interval: float       # s
    min_freq: float       # Hz
    max_freq: float
    avg_freq: float
    bandwidth: float
    avg_intensity: float
    bg_intensity: float
    area: float
    centroid_y: int
    top1: str | None
    top2: str | None


@dataclasses.dataclass
class DetectResult:
    path: pathlib.Path
    detections: list[Detection]
    csv_path: pathlib.Path | None
    output_dir: pathlib.Path
    duration: float
    samplerate: int
    params: dict


def resolved_cutoffs(animal: str, lo, hi) -> tuple[int, int]:
    """Return the (lo, hi) Hz that the engine will actually use, for display."""
    d_lo, d_hi = DEFAULT_CUTOFFS.get(animal, (0, 125000))
    lo = d_lo if lo in ("default", None) else int(lo)
    hi = d_hi if hi in ("default", None) else int(hi)
    return lo, hi


def _detections_from_recording(recording) -> list[Detection]:
    lov = recording.list_of_vocals
    out: list[Detection] = []
    if lov is None or lov.number_of_vocals == 0:
        return out
    vocals = sorted(lov.vocals_in_recording, key=lambda v: v.start)
    for i, v in enumerate(vocals, start=1):
        centroid_y = int(v.centroid[0]) if v.centroid is not None else 0
        out.append(Detection(
            index=i,
            bin_number=int(v.bin_number),
            start=float(v.start),
            end=float(v.end),
            duration_ms=float(v.duration),
            interval=float(v.interval) if v.interval is not None else 0.0,
            min_freq=float(v.min_freq),
            max_freq=float(v.max_freq),
            avg_freq=float(v.avg_freq),
            bandwidth=float(v.bandwidth),
            avg_intensity=float(v.avg_intensity) if v.avg_intensity is not None else float("nan"),
            bg_intensity=float(v.bg_intensity) if v.bg_intensity is not None else float("nan"),
            area=float(v.area) if v.area is not None else 0.0,
            centroid_y=centroid_y,
            top1=v.top1,
            top2=v.top2,
        ))
    return out


class _LogForwarder(logging.Handler):
    """Forward engine INFO log lines to a callback (for the GUI log panel)."""

    def __init__(self, callback):
        super().__init__(level=logging.INFO)
        self._cb = callback

    def emit(self, record):
        try:
            self._cb(record.getMessage())
        except Exception:
            pass


def run_detection(
    path: str | pathlib.Path,
    params: DetectParams,
    *,
    phase_cb=None,
    log_cb=None,
) -> DetectResult:
    """Run the full detect -> classify -> (segment) -> save pipeline for one file.

    ``phase_cb(str)`` is called with a coarse phase label; ``log_cb(str)`` receives
    the engine's own INFO log lines. Outputs (CSV, masks, serialized object) land in
    ``{stem}_outputs/`` next to the audio file, matching the upstream contract.
    """
    path = pathlib.Path(path)
    args = validate_arguments(params.to_args(path))

    root = logging.getLogger()
    handler = None
    prev_level = root.level
    if log_cb is not None:
        handler = _LogForwarder(log_cb)
        root.addHandler(handler)
        root.setLevel(logging.INFO)

    def phase(msg):
        if phase_cb is not None:
            phase_cb(msg)

    try:
        phase("Reading audio + spectrogram…")
        recording = Recording(recording_path=str(path), args=args)
        duration = float(recording.audio.audio_duration)
        samplerate = int(recording.audio.sampling_rate)

        phase("Detecting candidate vocalizations…")
        recording.identify_vocalizations()

        phase("Classifying (noise filter + syllable type)…")
        recording.classify_vocalizations()

        if params.segmenter:
            phase("Segmenting calls (SqueakOut)…")
        recording.segment_vocalizations()

        detections = _detections_from_recording(recording)

        phase("Writing CSV + outputs…")
        recording.save_outputs(validation_flag=params.validation)

        output_dir = pathlib.Path(get_output_directory_for_audio_file(str(path)))
        csv_path = output_dir / f"{path.stem}_stats.csv"
        return DetectResult(
            path=path,
            detections=detections,
            csv_path=csv_path if csv_path.exists() else None,
            output_dir=output_dir,
            duration=duration,
            samplerate=samplerate,
            params=dict(recording.params),
        )
    finally:
        if handler is not None:
            root.removeHandler(handler)
            root.setLevel(prev_level)


def run_detection_window(path, start_sec: float, length_sec: float,
                         params: DetectParams, *, phase_cb=None, log_cb=None) -> DetectResult:
    """Detect on just a ``[start_sec, start_sec+length_sec)`` window.

    Fast + interactive: writes the window to a temp clip, runs the same pipeline,
    then shifts detection times back to absolute file time. Outputs land in a
    temp dir (not next to the source), so this never clobbers a full-file CSV.
    """
    import tempfile
    import soundfile as sf

    path = pathlib.Path(path)
    info = sf.info(str(path))
    sr = info.samplerate
    start = max(0, min(int(round(start_sec * sr)), max(0, info.frames - 1)))
    n = min(int(round(length_sec * sr)), info.frames - start)
    x, _ = sf.read(str(path), start=start, frames=n, always_2d=True)
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="uvs_win_"))
    clip = tmp_dir / f"{path.stem}_win.wav"
    sf.write(str(clip), x, sr)

    result = run_detection(clip, params, phase_cb=phase_cb, log_cb=log_cb)
    off = start / sr
    shifted = []
    for d in result.detections:
        shifted.append(dataclasses.replace(d, start=d.start + off, end=d.end + off))
    return DetectResult(path=path, detections=shifted, csv_path=result.csv_path,
                        output_dir=result.output_dir, duration=result.duration,
                        samplerate=result.samplerate, params=result.params)


def summarize(detections: list[Detection], duration: float) -> dict:
    """Aggregate detection stats for the metric tiles."""
    n = len(detections)
    if n == 0:
        return {"n": 0, "per_min": 0.0, "mean_dur_ms": 0.0,
                "dominant_class": "—", "mean_peak_khz": 0.0}
    durs = np.array([d.duration_ms for d in detections], dtype=float)
    peaks = np.array([d.avg_freq for d in detections], dtype=float) / 1000.0
    labels = [d.top1 for d in detections if d.top1]
    dominant = max(set(labels), key=labels.count) if labels else "—"
    per_min = n / (duration / 60.0) if duration > 0 else 0.0
    return {
        "n": n,
        "per_min": per_min,
        "mean_dur_ms": float(durs.mean()),
        "dominant_class": dominant,
        "mean_peak_khz": float(peaks.mean()),
    }
