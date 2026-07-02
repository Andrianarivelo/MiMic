"""Reusable Qt widgets: the embedded spectrogram canvas + navigation toolbar."""

from __future__ import annotations

import pathlib

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

import vocalpy as voc

from . import plotting, theme
from .spectro import SpectSettings


class MplCanvas(FigureCanvasQTAgg):
    """A matplotlib canvas pre-styled for the dark theme."""

    def __init__(self):
        self.figure = Figure(figsize=(8, 5))
        self.figure.patch.set_facecolor(theme.BG)
        super().__init__(self.figure)


class DarkToolbar(NavigationToolbar2QT):
    """Matplotlib navigation toolbar restyled to fit the workstation theme."""

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.setStyleSheet(
            f"""
            QToolBar {{ background:{theme.BG_PANEL}; border:none; padding:2px; }}
            QToolButton {{ background:transparent; border:none; padding:4px; border-radius:4px; }}
            QToolButton:hover {{ background:{theme.BORDER}; }}
            QToolButton:checked {{ background:{theme.ACCENT}; }}
            QLabel {{ color:{theme.TEXT_DIM}; background:transparent; }}
            """
        )


class SpectrogramView(QWidget):
    """Canvas + toolbar bundle with convenience render / placeholder / save."""

    def __init__(self):
        super().__init__()
        self.canvas = MplCanvas()
        self.toolbar = DarkToolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        self.show_placeholder(
            "Select an audio file to preview its spectrogram\n\n"
            "File ▸ Load example  loads a built-in VocalPy clip"
        )

    def render(
        self,
        sound: voc.Sound | None,
        spect: voc.Spectrogram,
        settings: SpectSettings,
        title: str | None = None,
        time_offset: float = 0.0,
    ) -> None:
        plotting.draw_spectrogram(
            self.canvas.figure, sound, spect, settings, title, time_offset
        )
        self.canvas.draw_idle()

    def show_placeholder(self, message: str) -> None:
        plotting.draw_placeholder(self.canvas.figure, message)
        self.canvas.draw_idle()

    def save_figure(self, path: str | pathlib.Path, dpi: int = 200) -> None:
        self.canvas.figure.savefig(
            path, dpi=dpi, facecolor=self.canvas.figure.get_facecolor()
        )
