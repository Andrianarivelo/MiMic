"""QThread workers so audio I/O and spectrogram computation never block the UI.

* :class:`PreviewWorker` computes a single spectrogram for the live view. Each
  request carries a monotonically increasing ``generation`` token; the window
  ignores results whose token is stale (the user already selected another file).
* :class:`BatchWorker` walks a list of files, writing ``.npz`` and/or ``.png``
  outputs, emitting progress and honoring cooperative cancellation.
"""

from __future__ import annotations

import dataclasses
import pathlib
import traceback

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Signal

from . import plotting, spectro
from .spectro import SpectSettings


# --------------------------------------------------------------------------- #
# Preview
# --------------------------------------------------------------------------- #
class PreviewWorker(QThread):
    """Compute one spectrogram off the UI thread.

    Only a ``[start_sec, start_sec + length_sec]`` window of the file is read
    (unless ``whole`` is True), so previews on long, high-sample-rate recordings
    stay instant instead of loading the entire file.
    """

    done = Signal(int, object, object, dict)   # generation, Sound, Spectrogram, stats
    failed = Signal(int, str)                  # generation, message

    def __init__(
        self,
        path: pathlib.Path,
        settings: SpectSettings,
        generation: int,
        start_sec: float = 0.0,
        length_sec: float | None = None,
        whole: bool = False,
    ):
        super().__init__()
        self.path = pathlib.Path(path)
        self.settings = settings
        self.generation = generation
        self.start_sec = start_sec
        self.length_sec = None if whole else length_sec

    def run(self) -> None:  # noqa: D401  (Qt entry point)
        try:
            sound = spectro.read_sound_segment(
                self.path, self.start_sec, self.length_sec
            )
            spect = spectro.make_spectrogram(sound, self.settings)
            stats = spectro.spectrogram_stats(spect)
            self.done.emit(self.generation, sound, spect, stats)
        except Exception as exc:  # surface any failure to the UI
            self.failed.emit(self.generation, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class BatchConfig:
    output_dir: pathlib.Path
    save_npz: bool = True
    save_png: bool = True
    png_dpi: int = 150
    max_seconds: float = 0.0   # 0 = whole file; otherwise clip each file
    overwrite: bool = False    # if False, unique-suffix colliding output names


class BatchWorker(QThread):
    """Process many audio files into spectrogram files / images."""

    progress = Signal(int, int, str)     # index (1-based), total, file name
    file_ok = Signal(str, str)           # file name, outputs description
    file_failed = Signal(str, str)       # file name, error message
    finished_all = Signal(int, int, bool)  # n_ok, n_fail, cancelled

    def __init__(
        self,
        paths: list[pathlib.Path],
        settings: SpectSettings,
        config: BatchConfig,
    ):
        super().__init__()
        self.paths = [pathlib.Path(p) for p in paths]
        self.settings = settings
        self.config = config
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation (checked between files)."""
        self._cancelled = True

    def run(self) -> None:  # noqa: D401
        cfg = self.config
        try:
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.file_failed.emit(str(cfg.output_dir), f"cannot create output folder: {exc}")
            self.finished_all.emit(0, len(self.paths), False)
            return
        total = len(self.paths)
        n_ok = n_fail = 0

        used_stems: dict[str, int] = {}
        for i, path in enumerate(self.paths, start=1):
            if self._cancelled:
                break
            self.progress.emit(i, total, path.name)
            try:
                length = cfg.max_seconds if cfg.max_seconds > 0 else None
                sound = spectro.read_sound_segment(path, 0.0, length)
                spect = spectro.make_spectrogram(sound, self.settings)

                outputs = []
                stem = self._unique_stem(path.stem, used_stems)
                if cfg.save_npz:
                    npz_path = cfg.output_dir / f"{stem}.npz"
                    spect.write(npz_path)
                    outputs.append(npz_path.name)
                if cfg.save_png:
                    png_path = cfg.output_dir / f"{stem}.png"
                    self._render_png(sound, spect, png_path, title=path.name)
                    outputs.append(png_path.name)

                n_ok += 1
                self.file_ok.emit(path.name, ", ".join(outputs) or "(no output)")
            except Exception as exc:
                n_fail += 1
                tb = traceback.format_exc(limit=1).strip().splitlines()[-1]
                self.file_failed.emit(path.name, f"{type(exc).__name__}: {exc} [{tb}]")

        self.finished_all.emit(n_ok, n_fail, self._cancelled)

    def _unique_stem(self, stem: str, used: dict[str, int]) -> str:
        """Avoid silently overwriting outputs when two inputs share a stem.

        With ``overwrite`` enabled the original stem is always returned; the
        caller accepts that same-named inputs clobber each other.
        """
        if self.config.overwrite:
            return stem
        n = used.get(stem, 0)
        used[stem] = n + 1
        return stem if n == 0 else f"{stem}_{n + 1}"

    def _render_png(self, sound, spect, png_path: pathlib.Path, title: str) -> None:
        """Render a spectrogram PNG on a private Agg figure (thread-safe)."""
        fig = Figure(figsize=(11, 4.2))
        FigureCanvasAgg(fig)  # attach a canvas so savefig works
        try:
            plotting.draw_spectrogram(fig, sound, spect, self.settings, title=title)
            fig.savefig(
                png_path,
                dpi=self.config.png_dpi,
                facecolor=fig.get_facecolor(),
            )
        finally:
            fig.clear()  # release the figure's artists promptly


# --------------------------------------------------------------------------- #
# Gallery thumbnails
# --------------------------------------------------------------------------- #
class GalleryWorker(QThread):
    """Render a small spectrogram thumbnail per file, progressively.

    Each file's first ``thumb_seconds`` are read (so long recordings stay fast)
    and rendered to PNG bytes off the UI thread; the window turns them into
    pixmaps as they arrive.
    """

    thumb_ready = Signal(int, str, bytes, str)   # index, path, png bytes, subtitle
    thumb_failed = Signal(int, str, str)         # index, path, message
    progress = Signal(int, int)                  # done, total
    finished_all = Signal(int, int, bool)        # ok, fail, cancelled

    def __init__(self, items: list[pathlib.Path], settings: SpectSettings, thumb_seconds: float):
        super().__init__()
        self.items = [pathlib.Path(p) for p in items]
        self.settings = settings
        self.thumb_seconds = thumb_seconds
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: D401
        total = len(self.items)
        n_ok = n_fail = 0
        for i, path in enumerate(self.items):
            if self._cancelled:
                break
            try:
                info = spectro.sound_info(path)
                sub = f"{info.duration:.0f} s · {info.samplerate / 1000:.0f} kHz · {info.channels} ch"
                sound = spectro.read_sound_segment(path, 0.0, self.thumb_seconds)
                spect = spectro.make_spectrogram(sound, self.settings)
                png = plotting.render_thumbnail_png(spect, self.settings)
                n_ok += 1
                self.thumb_ready.emit(i, str(path), png, sub)
            except Exception as exc:
                n_fail += 1
                self.thumb_failed.emit(i, str(path), f"{type(exc).__name__}: {exc}")
            self.progress.emit(i + 1, total)
        self.finished_all.emit(n_ok, n_fail, self._cancelled)
