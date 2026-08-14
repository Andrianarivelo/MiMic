"""Reusable Qt widgets: metric tiles, the interactive spectrogram canvas (with a
hover crosshair readout and detection overlays), and the navigation toolbar."""

from __future__ import annotations

import pathlib

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from . import plotting, theme
from .spectro import DisplaySettings


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(figsize=(8, 5))
        self.figure.patch.set_facecolor(theme.BG)
        super().__init__(self.figure)


class DarkToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.setStyleSheet(
            f"""
            QToolBar {{ background:{theme.BG_PANEL}; border:none; padding:2px; }}
            QToolButton {{ background:transparent; border:none; padding:4px; border-radius:5px; }}
            QToolButton:hover {{ background:{theme.BG_ELEV2}; }}
            QToolButton:checked {{ background:{theme.ACCENT}; }}
            QLabel {{ color:{theme.TEXT_DIM}; background:transparent; }}
            """
        )


class MetricTile(QFrame):
    def __init__(self, key: str, unit: str = "", accent: bool = False, min_w: int = 96):
        super().__init__()
        self.setObjectName("Tile")
        if accent:
            self.setProperty("accent", "true")
        self.setMinimumWidth(min_w)
        v = QVBoxLayout(self)
        v.setContentsMargins(11, 6, 11, 7)
        v.setSpacing(1)
        self.key_lbl = QLabel(key.upper())
        self.key_lbl.setObjectName("TileKey")
        v.addWidget(self.key_lbl)
        row = QHBoxLayout()
        row.setSpacing(4)
        row.setContentsMargins(0, 0, 0, 0)
        self.val_lbl = QLabel("—")
        self.val_lbl.setObjectName("TileValAccent" if accent else "TileVal")
        row.addWidget(self.val_lbl)
        self.unit_lbl = QLabel(unit)
        self.unit_lbl.setObjectName("TileUnit")
        row.addWidget(self.unit_lbl)
        row.addStretch(1)
        v.addLayout(row)

    def set_value(self, value: str) -> None:
        self.val_lbl.setText(value)


class MetricsBar(QWidget):
    """Live detection-summary tiles + a cursor readout."""

    def __init__(self):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 2, 2, 4)
        row.setSpacing(8)
        self.calls = MetricTile("Vocalizations", "", accent=True)
        self.per_min = MetricTile("Calls / min", "")
        self.mean_dur = MetricTile("Mean duration", "ms")
        self.dominant = MetricTile("Top class", "")
        self.cursor = MetricTile("Cursor  ·  hover the spectrogram", min_w=250)
        for w in (self.calls, self.per_min, self.mean_dur, self.dominant):
            row.addWidget(w)
        row.addWidget(self.cursor, stretch=1)

    def set_detection_stats(self, s: dict) -> None:
        self.calls.set_value(str(s["n"]))
        self.per_min.set_value(f"{s['per_min']:.1f}")
        self.mean_dur.set_value(f"{s['mean_dur_ms']:.1f}")
        self.dominant.set_value(str(s["dominant_class"]))

    def clear(self) -> None:
        for t in (self.calls, self.per_min, self.mean_dur, self.dominant):
            t.set_value("—")
        self.set_cursor("—")

    def set_cursor(self, text: str) -> None:
        self.cursor.set_value(text)


class SpectrogramView(QWidget):
    """Metric tiles + canvas + toolbar with an interactive cursor readout."""

    def __init__(self):
        super().__init__()
        self.canvas = MplCanvas()
        self.toolbar = DarkToolbar(self.canvas, self)
        self.metrics = MetricsBar()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        layout.addWidget(self.metrics)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        self._im = None
        self._cross_v = None
        self._cross_h = None
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("axes_leave_event", self._on_mouse_leave)
        self.canvas.mpl_connect("figure_leave_event", self._on_mouse_leave)

        self.show_placeholder(
            "Add audio files, select one to preview, then Detect vocalizations\n\n"
            "Detected calls are boxed by predicted syllable class and exported to CSV"
        )

    def render(self, f, t, S, samples, sr, settings: DisplaySettings,
               detections=None, title=None) -> None:
        self._im = plotting.draw_spectrogram(
            self.canvas.figure, f, t, S, samples, sr, settings, detections, title)
        self._cross_v = self._cross_h = None
        self.canvas.draw_idle()

    def show_placeholder(self, message: str) -> None:
        self._im = None
        self._cross_v = self._cross_h = None
        plotting.draw_placeholder(self.canvas.figure, message)
        self.canvas.draw_idle()
        self.metrics.clear()

    def save_figure(self, path, dpi: int = 200) -> None:
        self.canvas.figure.savefig(path, dpi=dpi, facecolor=self.canvas.figure.get_facecolor())

    def _on_mouse_move(self, event) -> None:
        if self._im is None or event.inaxes is not self._im.axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        ax = self._im.axes
        val = self._im.get_cursor_data(event)
        if val is None:
            return
        self.metrics.set_cursor(
            f"t {event.xdata:.3f} s   ·   {event.ydata / 1000.0:.1f} kHz   ·   {val:+.1f} dB")
        if self._cross_v is None:
            self._cross_v = ax.axvline(event.xdata, color=theme.ACCENT, lw=0.8, ls="--", alpha=0.85)
            self._cross_h = ax.axhline(event.ydata, color=theme.ACCENT, lw=0.8, ls="--", alpha=0.85)
        else:
            self._cross_v.set_xdata([event.xdata, event.xdata])
            self._cross_h.set_ydata([event.ydata, event.ydata])
            self._cross_v.set_visible(True)
            self._cross_h.set_visible(True)
        self.canvas.draw_idle()

    def _on_mouse_leave(self, _event) -> None:
        changed = False
        if self._cross_v is not None:
            self._cross_v.set_visible(False)
            self._cross_h.set_visible(False)
            changed = True
        self.metrics.set_cursor("—")
        if changed:
            self.canvas.draw_idle()
