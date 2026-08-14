"""Detections table: one row per detected vocalization, class-colored, clickable."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem,
)

from . import theme
from .engine import class_color

COLUMNS = ["#", "Start (s)", "End (s)", "Dur (ms)", "Freq (kHz)", "Bandwidth (kHz)", "Class", "2nd"]


class DetectionsTable(QTableWidget):
    """Shows detected vocalizations; emits the start time when a row is chosen."""

    row_activated = Signal(float)   # start time (s)

    def __init__(self):
        super().__init__(0, len(COLUMNS))
        self.setHorizontalHeaderLabels(COLUMNS)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.verticalHeader().setVisible(False)
        self.setWordWrap(False)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._starts: list[float] = []
        self.cellClicked.connect(self._on_cell_clicked)
        self.setStyleSheet(
            f"""
            QTableWidget {{
                background:{theme.BG_PLOT}; border:1px solid {theme.BORDER};
                border-radius:8px; gridline-color:{theme.BORDER};
                alternate-background-color:{theme.BG_PANEL};
                selection-background-color:{theme.ACCENT_DEEP}; selection-color:{theme.ACCENT_INK};
            }}
            QTableWidget::item {{ padding:4px 8px; }}
            QHeaderView::section {{
                background:{theme.BG_ELEV}; color:{theme.TEXT_DIM};
                border:none; border-bottom:1px solid {theme.BORDER};
                padding:6px 8px; font-weight:700; font-size:11px;
            }}
            """
        )

    def set_detections(self, detections) -> None:
        self._starts = [d.start for d in detections]
        self.setRowCount(len(detections))
        for r, d in enumerate(detections):
            vals = [
                str(d.index),
                f"{d.start:.3f}", f"{d.end:.3f}", f"{d.duration_ms:.1f}",
                f"{d.min_freq / 1000:.1f}–{d.max_freq / 1000:.1f}",
                f"{d.bandwidth / 1000:.1f}",
                d.top1 or "—", d.top2 or "—",
            ]
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setForeground(QColor(theme.TEXT_FAINT))
                if c in (1, 2, 3, 5):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if c == 6:
                    item.setForeground(QColor(class_color(d.top1)))
                    f = item.font(); f.setBold(True); item.setFont(f)
                self.setItem(r, c, item)

    def clear_detections(self) -> None:
        self.setRowCount(0)
        self._starts = []

    def _on_cell_clicked(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._starts):
            self.row_activated.emit(self._starts[row])
