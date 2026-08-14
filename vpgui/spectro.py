"""Audio discovery, windowed reading, and display-spectrogram helpers.

No Qt here. Spectrograms are computed with the *same* routine the detection
engine uses (:func:`vocalpy.utils.signal_processing.compute_spectrogram`) so what
you see in the preview matches what the detector analyzes. Purely-display touches
(background denoise, contrast limits) live here too.
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib
from typing import Iterable

import numpy as np
import soundfile

from vocalpy.utils.signal_processing import compute_spectrogram
from vocalpy.utils.io import read_yaml
from vocalpy.configs import configs as _cfg_mod

AUDIO_EXTENSIONS = (".wav", ".WAV", ".flac", ".FLAC", ".ogg", ".aiff", ".aif")

COLORMAPS = [
    "magma", "inferno", "viridis", "plasma", "cividis",
    "turbo", "gray", "bone", "afmhot",
]


@dataclasses.dataclass
class DisplaySettings:
    """Display-only options for the spectrogram preview (never affect detection)."""

    colormap: str = "magma"
    dynamic_range_db: float = 55.0
    denoise: bool = True
    show_waveform: bool = True


@dataclasses.dataclass
class AudioInfo:
    samplerate: int
    channels: int
    frames: int
    duration: float
    subtype: str = ""

    @property
    def nyquist(self) -> float:
        return self.samplerate / 2.0


# --------------------------------------------------------------------------- #
# Discovery + metadata
# --------------------------------------------------------------------------- #
def find_audio_files(folder, recursive: bool = True,
                     extensions: Iterable[str] = AUDIO_EXTENSIONS) -> list[pathlib.Path]:
    folder = pathlib.Path(folder)
    exts = {e.lower() for e in extensions}
    globber = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in globber if p.is_file() and p.suffix.lower() in exts)


def sound_info(path) -> AudioInfo:
    info = soundfile.info(str(path))
    return AudioInfo(
        samplerate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
        duration=float(info.frames) / float(info.samplerate),
        subtype=str(info.subtype),
    )


def read_window(path, start_sec: float = 0.0, length_sec: float | None = None):
    """Read a mono window ``[start_sec, start_sec+length_sec)`` -> (samples, sr)."""
    info = sound_info(path)
    sr = info.samplerate
    if length_sec is None:
        x, _ = soundfile.read(str(path), always_2d=True)
    else:
        start = max(0, min(int(round(start_sec * sr)), max(0, info.frames - 1)))
        n = int(round(length_sec * sr))
        n = min(n, info.frames - start)
        if n <= 0:
            start, n = 0, min(info.frames, int(round(length_sec * sr)))
        x, _ = soundfile.read(str(path), start=start, frames=n, always_2d=True)
    return x[:, 0], sr


# --------------------------------------------------------------------------- #
# Spectrogram (engine-consistent) + display transforms
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _pipeline_params() -> dict:
    import os
    yml = os.path.join(os.path.dirname(_cfg_mod.__file__), "pipelines_parameters.yml")
    return read_yaml(yml)["pipelines"]


def spectrogram_params(animal: str) -> dict:
    """Spectrogram kwargs (nfft/window/cutoffs) the engine uses for ``animal``."""
    p = _pipeline_params().get(animal, _pipeline_params()["mouse"])
    return {
        "window_type": p["window_type"],
        "window_size": p["window_size"],
        "noverlap": p["noverlap"],
        "nfft": p["nfft"],
        "lower_frequency_cutoff": p["lower_frequency_cutoff"],
        "higher_frequency_cutoff": p["higher_frequency_cutoff"],
    }


def compute_display_spectrogram(samples, sr, animal="mouse",
                                lower_hz=None, higher_hz=None):
    """Return (f_hz, t_s, S_db) for ``samples`` using the engine's spectrogram."""
    kwargs = spectrogram_params(animal)
    if lower_hz is not None:
        kwargs["lower_frequency_cutoff"] = int(lower_hz)
    if higher_hz is not None:
        kwargs["higher_frequency_cutoff"] = int(higher_hz)
    f, t, S = compute_spectrogram(samples=samples, fs=sr, **kwargs)
    return f, t, S


def to_display_db(S_db: np.ndarray, denoise: bool = False) -> np.ndarray:
    """Optionally subtract each frequency row's stationary background (dB)."""
    data = S_db
    if denoise and data.size:
        finite = np.where(np.isfinite(data), data, np.nan)
        with np.errstate(all="ignore"):
            base = np.nanmedian(finite, axis=1, keepdims=True)
        base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
        data = data - base
    return data


def display_limits(data_db: np.ndarray, dynamic_range_db: float,
                   denoise: bool) -> tuple[float, float]:
    finite = data_db[np.isfinite(data_db)]
    if not finite.size:
        return -1.0, 0.0
    vmax = float(np.nanpercentile(finite, 99.7))
    if denoise:
        vmin = max(float(np.nanpercentile(finite, 66.0)), vmax - float(dynamic_range_db))
        if vmax - vmin < 8.0:
            vmin = vmax - 8.0
    else:
        vmin = vmax - float(dynamic_range_db)
    return vmin, vmax


# --------------------------------------------------------------------------- #
# Call-rich window seeking (for gallery thumbnails + preview jump)
# --------------------------------------------------------------------------- #
def find_active_window(path, window_sec: float = 4.0, n_probes: int = 7,
                       band: tuple[float, float] = (30_000.0, 110_000.0)) -> float:
    """Offset (s) of the most vocal ``window_sec`` slice; 0.0 on short/error."""
    from scipy.signal import stft
    try:
        info = sound_info(path)
    except Exception:
        return 0.0
    dur, sr = info.duration, info.samplerate
    if dur <= window_sec or n_probes < 1:
        return 0.0
    probe_sec = min(1.5, window_sec)
    span = max(0.0, dur - window_sec)
    offsets = [span * k / (n_probes - 1) for k in range(n_probes)] if n_probes > 1 else [0.0]
    best_off, best_score = 0.0, -1.0
    for off in offsets:
        try:
            x, _ = soundfile.read(str(path), start=int(off * sr),
                                  frames=int(probe_sec * sr), dtype="float32")
        except Exception:
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        if x.size < 512:
            continue
        f, _t, Z = stft(x, fs=sr, nperseg=1024, noverlap=512)
        P = np.abs(Z)
        sel = (f >= band[0]) & (f <= band[1])
        if not sel.any():
            continue
        Pb = P[sel]
        score = float(((Pb.max(0) / (np.median(Pb, 0) + 1e-9)) > 8.0).sum())
        if score > best_score:
            best_score, best_off = score, off
    return best_off
