"""QThread workers so audio I/O, spectrogram computation, and the (heavy) USV
detection pipeline never block the UI.

* :class:`PreviewWorker` computes a display spectrogram for one window.
* :class:`DetectWorker` runs the full detection pipeline for one file.
* :class:`BatchDetectWorker` detects a list of files, writing each CSV.
* :class:`GalleryWorker` renders a spectrogram thumbnail per file.
"""

from __future__ import annotations

import pathlib

from PySide6.QtCore import QThread, Signal

from . import engine, plotting, spectro
from .engine import DetectParams
from .spectro import DisplaySettings


class PreviewWorker(QThread):
    """Compute a windowed display spectrogram off the UI thread."""

    done = Signal(int, object, object, object, object, int, dict)  # gen,f,t,S,samples,sr,stats
    failed = Signal(int, str)

    def __init__(self, path, generation, start_sec, length_sec, whole,
                 animal, lower_hz, higher_hz):
        super().__init__()
        self.path = pathlib.Path(path)
        self.generation = generation
        self.start_sec = start_sec
        self.length_sec = None if whole else length_sec
        self.animal = animal
        self.lower_hz = lower_hz
        self.higher_hz = higher_hz

    def run(self):
        try:
            samples, sr = spectro.read_window(self.path, self.start_sec, self.length_sec)
            if samples.size == 0:
                raise ValueError("empty audio (0 samples) — file may be truncated")
            f, t, S = spectro.compute_display_spectrogram(
                samples, sr, self.animal, self.lower_hz, self.higher_hz)
            offset = 0.0 if self.length_sec is None else float(self.start_sec)
            stats = {"shape": (int(S.shape[0]), int(S.shape[1])), "sr": sr,
                     "duration": len(samples) / sr, "offset": offset}
            self.done.emit(self.generation, f, t + offset, S, samples, sr, stats)
        except Exception as exc:
            self.failed.emit(self.generation, f"{type(exc).__name__}: {exc}")


class DetectWorker(QThread):
    """Run the detect -> classify -> (segment) -> save pipeline for one file."""

    phase = Signal(str)
    log = Signal(str)
    done = Signal(object)     # engine.DetectResult
    failed = Signal(str)

    def __init__(self, path, params: DetectParams, window=None):
        super().__init__()
        self.path = pathlib.Path(path)
        self.params = params
        self.window = window   # None -> whole file; else (start_sec, length_sec)

    def run(self):
        try:
            if self.window is None:
                result = engine.run_detection(
                    self.path, self.params,
                    phase_cb=self.phase.emit, log_cb=self.log.emit)
            else:
                start, length = self.window
                result = engine.run_detection_window(
                    self.path, start, length, self.params,
                    phase_cb=self.phase.emit, log_cb=self.log.emit)
            self.done.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class BatchDetectWorker(QThread):
    """Detect a list of files, writing each ``*_stats.csv``."""

    progress = Signal(int, int, str)     # index (1-based), total, file name
    file_ok = Signal(str, int, str)      # name, n_calls, csv_path
    file_failed = Signal(str, str)
    phase = Signal(str)
    finished_all = Signal(int, int, bool)  # n_ok, n_fail, cancelled

    def __init__(self, paths, params: DetectParams):
        super().__init__()
        self.paths = [pathlib.Path(p) for p in paths]
        self.params = params
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        total = len(self.paths)
        n_ok = n_fail = 0
        for i, path in enumerate(self.paths, start=1):
            if self._cancelled:
                break
            self.progress.emit(i, total, path.name)
            try:
                self.phase.emit(f"[{i}/{total}] {path.name}")
                result = engine.run_detection(path, self.params,
                                              phase_cb=lambda m: None, log_cb=None)
                n_ok += 1
                csv = str(result.csv_path) if result.csv_path else "(no csv)"
                self.file_ok.emit(path.name, len(result.detections), csv)
            except Exception as exc:
                n_fail += 1
                self.file_failed.emit(path.name, f"{type(exc).__name__}: {exc}")
        self.finished_all.emit(n_ok, n_fail, self._cancelled)


class GalleryWorker(QThread):
    """Render a spectrogram thumbnail per file (seeking a call-rich window)."""

    thumb_ready = Signal(int, str, bytes, str, float)   # idx, path, png, subtitle, offset
    thumb_failed = Signal(int, str, str)
    progress = Signal(int, int)
    finished_all = Signal(int, int, bool)

    def __init__(self, items, display: DisplaySettings, thumb_seconds, animal):
        super().__init__()
        self.items = [pathlib.Path(p) for p in items]
        self.display = display
        self.thumb_seconds = thumb_seconds
        self.animal = animal
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        total = len(self.items)
        n_ok = n_fail = 0
        for i, path in enumerate(self.items):
            if self._cancelled:
                break
            try:
                info = spectro.sound_info(path)
                offset = spectro.find_active_window(path, self.thumb_seconds)
                where = f" · @{offset:.0f}s" if offset > 0 else ""
                sub = (f"{info.duration:.0f} s · {info.samplerate / 1000:.0f} kHz · "
                       f"{info.channels} ch{where}")
                samples, sr = spectro.read_window(path, offset, self.thumb_seconds)
                f, t, S = spectro.compute_display_spectrogram(samples, sr, self.animal)
                png = plotting.render_thumbnail_png(f, t, S, self.display)
                n_ok += 1
                self.thumb_ready.emit(i, str(path), png, sub, offset)
            except Exception as exc:
                n_fail += 1
                self.thumb_failed.emit(i, str(path), f"{type(exc).__name__}: {exc}")
            self.progress.emit(i + 1, total)
        self.finished_all.emit(n_ok, n_fail, self._cancelled)
