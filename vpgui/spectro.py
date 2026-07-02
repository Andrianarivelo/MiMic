"""Pure :mod:`vocalpy` logic: audio discovery, loading and spectrogram dispatch.

This module deliberately contains **no Qt and no pyplot** so it can be unit
tested headless and reused by both the preview and batch workers.

Two upstream quirks in vocalpy 0.11.0 are worked around here so all three
methods "just work":

1. :func:`vocalpy.spectrogram` forwards ``n_fft`` / ``hop_length`` positionally
   into every backend.  For ``method='soundsig-spectro'`` that backend does
   **not** accept those names, so the call raises
   ``TypeError: multiple values for argument 'spec_sample_rate'``.
2. :func:`vocalpy.spectral.soundsig_spectro` with its default ``scale=True``
   rebuilds the ``Sound`` with ``path=sound.path`` -- but :class:`vocalpy.Sound`
   has no ``path`` attribute, so it raises ``AttributeError``.

We therefore dispatch the soundsig method directly to the spectral function,
pre-scaling the signal to int16 ourselves and passing ``scale=False`` (which
both reproduces soundsig's int16 assumption and avoids the broken code path).
``librosa-db`` / ``sat-multitaper`` are routed through ``voc.spectrogram``.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Iterable

import numpy as np
import vocalpy as voc

# Methods exposed in the UI.  Keys are user-facing labels.
METHODS = ("librosa-db", "sat-multitaper", "soundsig-spectro")

# Audio extensions we scan for when adding a folder.  soundfile reads the first
# group; vocalpy adds ``.cbin`` (Bengalese-finch / evTAF) via its evfuncs vendor.
AUDIO_EXTENSIONS = (
    ".wav", ".flac", ".ogg", ".oga", ".aiff", ".aif",
    ".aifc", ".w64", ".mat5", ".cbin",
)

SOUNDSIG_SCALE_VAL = 2 ** 15  # soundsig assumes int16-scaled audio

# Built-in vocalpy example clips (downloaded + cached by pooch on first use).
EXAMPLE_CLIPS = (
    "bells.wav",
    "samba.wav",
    "bl26lb16.wav",
    "BM003.wav",
    "deermouse-go.wav",
    "flashcam.wav",
    "simple.wav",
)


@dataclasses.dataclass
class SpectSettings:
    """All parameters needed to compute and display a spectrogram."""

    # --- computation ---
    method: str = "librosa-db"
    n_fft: int = 512
    hop_length: int = 64
    to_mono: bool = True

    # --- soundsig-spectro specific ---
    spec_sample_rate: int = 1000
    freq_spacing: int = 50
    min_freq: int = 0
    max_freq: int = 10000
    nstd: int = 6

    # --- display only (do not require recompute) ---
    colormap: str = "magma"
    dynamic_range_db: float = 80.0
    show_waveform: bool = True
    limit_freq: bool = False
    flim_min_khz: float = 0.0
    flim_max_khz: float = 15.0


# --------------------------------------------------------------------------- #
# Discovery & loading
# --------------------------------------------------------------------------- #
def find_audio_files(
    folder: str | pathlib.Path,
    recursive: bool = True,
    extensions: Iterable[str] = AUDIO_EXTENSIONS,
) -> list[pathlib.Path]:
    """Return a sorted list of audio files inside ``folder``.

    Matching is case-insensitive on the extension.
    """
    folder = pathlib.Path(folder)
    exts = {e.lower() for e in extensions}
    globber = folder.rglob("*") if recursive else folder.glob("*")
    found = [
        p for p in globber
        if p.is_file() and p.suffix.lower() in exts
    ]
    return sorted(found)


def read_sound(path: str | pathlib.Path) -> voc.Sound:
    """Read a whole audio file into a :class:`vocalpy.Sound`."""
    return voc.Sound.read(pathlib.Path(path))


@dataclasses.dataclass
class AudioInfo:
    """Lightweight metadata read without loading the audio samples."""

    samplerate: int
    channels: int
    frames: int
    duration: float          # seconds
    subtype: str = ""

    @property
    def nyquist(self) -> float:
        return self.samplerate / 2.0


def sound_info(path: str | pathlib.Path) -> AudioInfo:
    """Return audio metadata cheaply (no sample data is read).

    Falls back to a full read for formats :mod:`soundfile` cannot probe
    (e.g. ``.cbin``), which are small enough that this is fine.
    """
    import soundfile

    path = pathlib.Path(path)
    try:
        info = soundfile.info(str(path))
        return AudioInfo(
            samplerate=int(info.samplerate),
            channels=int(info.channels),
            frames=int(info.frames),
            duration=float(info.frames) / float(info.samplerate),
            subtype=str(info.subtype),
        )
    except Exception:
        snd = read_sound(path)
        return AudioInfo(
            samplerate=int(snd.samplerate),
            channels=int(snd.channels),
            frames=int(snd.samples),
            duration=float(snd.duration),
        )


def read_sound_segment(
    path: str | pathlib.Path,
    start_sec: float = 0.0,
    length_sec: float | None = None,
) -> voc.Sound:
    """Read only a ``length_sec`` window starting at ``start_sec`` seconds.

    This is what makes the app usable on long, high-rate recordings: instead of
    loading hundreds of MB and computing a multi-GB spectrogram, we read just the
    requested window via ``soundfile``'s ``start`` / ``frames`` arguments
    (forwarded through :meth:`vocalpy.Sound.read`).

    ``length_sec=None`` (or a window covering the whole file) reads everything.
    """
    path = pathlib.Path(path)
    if length_sec is None:
        return read_sound(path)

    try:
        info = sound_info(path)
        sr = info.samplerate
        start_frame = max(0, int(round(start_sec * sr)))
        start_frame = min(start_frame, max(0, info.frames - 1))
        n_frames = int(round(length_sec * sr))
        n_frames = min(n_frames, info.frames - start_frame)
        if n_frames <= 0:
            n_frames = min(info.frames, int(round(length_sec * sr)))
            start_frame = 0
        return voc.Sound.read(path, start=start_frame, frames=n_frames)
    except TypeError:
        # Backend (e.g. cbin) does not support start/frames: read + clip.
        snd = read_sound(path)
        stop = None if length_sec is None else start_sec + length_sec
        return snd.clip(float(start_sec), stop if stop is None else float(stop))


# --------------------------------------------------------------------------- #
# Spectrogram dispatch
# --------------------------------------------------------------------------- #
def _soundsig_scaled(
    sound: voc.Sound,
    spec_sample_rate: int = 1000,
    freq_spacing: int = 50,
    min_freq: int = 0,
    max_freq: int = 10000,
    nstd: int = 6,
) -> voc.Spectrogram:
    """Robust wrapper around :func:`vocalpy.spectral.soundsig_spectro`.

    We pre-scale the audio to int16 and pass ``scale=False`` to reproduce
    soundsig's behavior while avoiding the upstream ``Sound.path`` crash
    (see module docstring). Usable as a :class:`vocalpy.SpectrogramMaker`
    callback as well.
    """
    # Clip before casting so non-normalized / clipping audio cannot wrap around
    # the int16 range (e.g. a +1.0 peak * 32768 -> 32768 would wrap to -32768).
    scaled_f = np.clip(sound.data * SOUNDSIG_SCALE_VAL, -32768.0, 32767.0)
    scaled = voc.Sound(
        data=scaled_f.astype(np.int16),
        samplerate=sound.samplerate,
    )
    # soundsig does 20*log10(|spec|); silent bins give log10(0) -> -inf.
    # That is expected, so suppress the (upstream) divide-by-zero warning.
    with np.errstate(divide="ignore"):
        return voc.spectral.soundsig_spectro(
            scaled,
            scale=False,
            spec_sample_rate=int(spec_sample_rate),
            freq_spacing=int(freq_spacing),
            min_freq=int(min_freq),
            max_freq=int(max_freq),
            nstd=int(nstd),
        )


def make_spectrogram(sound: voc.Sound, settings: SpectSettings) -> voc.Spectrogram:
    """Compute a :class:`vocalpy.Spectrogram` from ``sound`` using ``settings``.

    Multi-channel audio is collapsed to mono when ``settings.to_mono`` is set;
    otherwise only the first channel is used for the (single-channel) display
    representation downstream.
    """
    if sound.samples == 0:
        raise ValueError("empty audio (0 samples) — file may be corrupt or truncated")

    if settings.to_mono and sound.channels > 1:
        sound = sound.to_mono()

    if settings.method == "soundsig-spectro":
        spect = _soundsig_scaled(
            sound,
            spec_sample_rate=settings.spec_sample_rate,
            freq_spacing=settings.freq_spacing,
            min_freq=settings.min_freq,
            max_freq=settings.max_freq,
            nstd=settings.nstd,
        )
    else:
        spect = voc.spectrogram(
            sound,
            n_fft=int(settings.n_fft),
            hop_length=int(settings.hop_length),
            method=settings.method,
        )
    return spect


def first_channel(spect: voc.Spectrogram) -> voc.Spectrogram:
    """Return a single-channel view of ``spect`` (vocalpy plotting needs 2-D)."""
    if spect.data.shape[0] > 1:
        return spect[0]
    return spect


def make_spectrogram_maker(settings: SpectSettings) -> voc.SpectrogramMaker:
    """Build a :class:`vocalpy.SpectrogramMaker` matching ``settings``.

    Provided for callers that want to drive vocalpy's own batch pipeline
    (``maker.make([...])``).  The GUI's batch worker iterates per-file instead so
    it can report progress and honor cancellation, but this keeps the canonical
    vocalpy entry point one call away.
    """
    if settings.method == "soundsig-spectro":
        return voc.SpectrogramMaker(
            callback=_soundsig_scaled,
            params={
                "spec_sample_rate": int(settings.spec_sample_rate),
                "freq_spacing": int(settings.freq_spacing),
                "min_freq": int(settings.min_freq),
                "max_freq": int(settings.max_freq),
                "nstd": int(settings.nstd),
            },
        )
    return voc.SpectrogramMaker(
        callback=voc.spectrogram,
        params={
            "n_fft": int(settings.n_fft),
            "hop_length": int(settings.hop_length),
            "method": settings.method,
        },
    )


def estimate_spectrogram_cells(
    samplerate: int, n_samples: int, settings: SpectSettings
) -> float:
    """Rough freq×time cell count a spectrogram would have, for a memory guard."""
    if settings.method == "soundsig-spectro":
        cols = (n_samples / max(1, samplerate)) * max(1, settings.spec_sample_rate)
        nfreq = max(1.0, (settings.max_freq - settings.min_freq) / max(1, settings.freq_spacing))
    else:
        cols = n_samples / max(1, settings.hop_length)
        nfreq = settings.n_fft // 2 + 1
    return float(cols) * float(nfreq)


def spectrogram_stats(spect: voc.Spectrogram) -> dict:
    """Small summary used in the UI/log."""
    data = spect.data
    finite = data[np.isfinite(data)]
    vmin = float(finite.min()) if finite.size else float("nan")
    vmax = float(finite.max()) if finite.size else float("nan")
    return {
        "shape": tuple(data.shape),
        "f_min": float(spect.frequencies.min()),
        "f_max": float(spect.frequencies.max()),
        "t_min": float(spect.times.min()),
        "t_max": float(spect.times.max()),
        "vmin": vmin,
        "vmax": vmax,
    }
