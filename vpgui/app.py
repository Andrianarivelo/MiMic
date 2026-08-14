"""VocalPy USV Workbench - main window and entry point.

A dark "scientific workstation" desktop app (PySide6) that drives the
``gumadeiras/vocalpy`` detection pipeline: browse recordings, preview
band-limited spectrograms, detect + classify (+ segment) ultrasonic
vocalizations, inspect them as boxed overlays and a results table, and export
the per-file ``*_stats.csv``.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

from PySide6.QtCore import Qt, QRectF, QThread
from PySide6.QtGui import (
    QAction, QBrush, QColor, QGuiApplication, QLinearGradient, QPainter,
    QPainterPath, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)

from . import engine, spectro, theme
from .engine import DetectParams
from .gallery import GalleryView
from .results import DetectionsTable
from .spectro import DisplaySettings
from .widgets import SpectrogramView
from .workers import BatchDetectWorker, DetectWorker, GalleryWorker, PreviewWorker

ROLE_PATH = Qt.UserRole
MAX_PREVIEW_SECONDS = 60.0   # guard: cap a single preview window


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VocalPy USV Workbench")
        self.resize(1500, 950)

        # --- state ---
        self.current_path: pathlib.Path | None = None
        self.current_info: spectro.AudioInfo | None = None
        self.window_offset: float = 0.0
        self._cur = None                      # cached preview arrays for re-render
        self._detections_by_path: dict[str, list] = {}
        self._preview_gen = 0
        self._pending_path: pathlib.Path | None = None
        self._live_workers: list[QThread] = []
        self._preview_worker: PreviewWorker | None = None
        self._detect_worker: DetectWorker | None = None
        self._batch_worker: BatchDetectWorker | None = None
        self._gallery_worker: GalleryWorker | None = None
        self._gallery_offsets: dict[str, float] = {}

        self._build_ui()
        self._build_menu()
        self._connect()
        self._on_species_changed()
        self._configure_window_controls()
        self._update_counts()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 10, 12, 8)
        outer.setSpacing(8)
        outer.addWidget(self._build_header())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_column())
        splitter.addWidget(self._build_right_column())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 1050])
        outer.addWidget(splitter, stretch=1)

        self.setStatusBar(self.statusBar())
        self.statusBar().showMessage(
            f"vocalpy engine {getattr(__import__('vocalpy'), '__version__', '?')}  ·  ready")

    # ---- header ----
    def _build_header(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("HeaderBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(16, 11, 14, 11)
        row.setSpacing(14)

        logo = QLabel()
        logo.setObjectName("LogoGlyph")
        logo.setPixmap(self._make_logo(46))
        logo.setFixedSize(46, 46)
        row.addWidget(logo)

        left = QVBoxLayout()
        left.setSpacing(1)
        title = QLabel("VocalPy <span style='color:%s'>USV Workbench</span>" % theme.ACCENT)
        title.setObjectName("Heading")
        title.setTextFormat(Qt.RichText)
        sub = QLabel("DETECT · CLASSIFY · SEGMENT ULTRASONIC VOCALIZATIONS → CSV")
        sub.setObjectName("SubHeading")
        left.addWidget(title)
        left.addWidget(sub)
        row.addLayout(left)
        row.addStretch(1)

        self.chip_file = self._make_chip("File")
        self.chip_rate = self._make_chip("Sample rate")
        self.chip_dur = self._make_chip("Duration")
        self.chip_file.value.setText("no file")
        for chip in (self.chip_file, self.chip_rate, self.chip_dur):
            row.addWidget(chip)
        return bar

    def _make_logo(self, size: int) -> QPixmap:
        ratio = 2
        pm = QPixmap(size * ratio, size * ratio)
        pm.setDevicePixelRatio(ratio)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        path = QPainterPath()
        path.addRoundedRect(QRectF(1, 1, size - 2, size - 2), 11, 11)
        grad = QLinearGradient(0, 0, 0, size)
        grad.setColorAt(0.0, QColor("#123246"))
        grad.setColorAt(1.0, QColor("#0a141d"))
        p.fillPath(path, QBrush(grad))
        p.setPen(QPen(QColor(theme.BORDER_GLOW), 1.2))
        p.drawPath(path)
        bars = [0.32, 0.62, 0.45, 0.88, 0.55, 0.72, 0.38]
        n = len(bars)
        gap, pad = 2.4, 9.0
        bw = (size - 2 * pad - (n - 1) * gap) / n
        base = size - pad
        cols = [QColor(theme.ACCENT_BLUE), QColor(theme.ACCENT), QColor(theme.ACCENT_TEAL)]
        for i, h in enumerate(bars):
            x = pad + i * (bw + gap)
            top = base - h * (size - 2 * pad)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(cols[i % len(cols)]))
            p.drawRoundedRect(QRectF(x, top, bw, base - top), 1.4, 1.4)
        p.end()
        return pm

    def _make_chip(self, key: str) -> QFrame:
        chip = QFrame()
        chip.setObjectName("Chip")
        v = QVBoxLayout(chip)
        v.setContentsMargins(12, 6, 12, 7)
        v.setSpacing(0)
        k = QLabel(key.upper())
        k.setObjectName("ChipKey")
        val = QLabel("—")
        val.setObjectName("ChipVal")
        v.addWidget(k)
        v.addWidget(val)
        chip.value = val
        return chip

    # ---- left column ----
    def _build_left_column(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 2, 0)
        v.setSpacing(7)
        v.addWidget(self._build_files_group())
        v.addWidget(self._build_analysis_group())
        v.addWidget(self._build_window_group())
        v.addWidget(self._build_display_group())
        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(410)
        scroll.setMaximumWidth(480)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    def _build_files_group(self) -> QGroupBox:
        box = QGroupBox("Audio files")
        v = QVBoxLayout(box)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.setMinimumHeight(140)
        v.addWidget(self.file_list)
        self.count_label = QLabel("0 files")
        self.count_label.setObjectName("Hint")
        v.addWidget(self.count_label)
        row1 = QHBoxLayout()
        self.btn_add_files = QPushButton("Add files…")
        self.btn_add_folder = QPushButton("Add folder…")
        row1.addWidget(self.btn_add_files)
        row1.addWidget(self.btn_add_folder)
        v.addLayout(row1)
        row3 = QHBoxLayout()
        self.btn_remove = QPushButton("Remove selected")
        self.btn_clear = QPushButton("Clear all")
        self.btn_clear.setObjectName("Danger")
        row3.addWidget(self.btn_remove)
        row3.addWidget(self.btn_clear)
        v.addLayout(row3)
        self.recursive_check = QCheckBox("Scan sub-folders recursively")
        self.recursive_check.setChecked(True)
        v.addWidget(self.recursive_check)
        return box

    def _build_analysis_group(self) -> QGroupBox:
        box = QGroupBox("Detection pipeline")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self.species_combo = QComboBox()
        self.species_combo.addItems(list(engine.SPECIES))
        form.addRow("Species", self.species_combo)

        self.bin_spin = QSpinBox()
        self.bin_spin.setRange(5, 600)
        self.bin_spin.setValue(60)
        self.bin_spin.setSuffix(" s")
        self.bin_spin.setToolTip("Chunk size for parallel processing")
        form.addRow("Bin size", self.bin_spin)

        self.auto_band_check = QCheckBox("Auto band (species default)")
        self.auto_band_check.setChecked(True)
        form.addRow("", self.auto_band_check)

        band_row = QHBoxLayout()
        self.flo_spin = QDoubleSpinBox()
        self.flo_spin.setRange(0.0, 400.0)
        self.flo_spin.setSuffix(" kHz")
        self.fhi_spin = QDoubleSpinBox()
        self.fhi_spin.setRange(0.1, 400.0)
        self.fhi_spin.setSuffix(" kHz")
        band_row.addWidget(self.flo_spin)
        band_row.addWidget(QLabel("→"))
        band_row.addWidget(self.fhi_spin)
        wrap = QWidget(); wrap.setLayout(band_row); band_row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Band", wrap)

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(-1, 64)
        self.threads_spin.setValue(-1)
        self.threads_spin.setSpecialValueText("auto")
        form.addRow("Threads", self.threads_spin)

        self.segmenter_check = QCheckBox("SqueakOut neural segmentation")
        form.addRow("", self.segmenter_check)
        self.validation_check = QCheckBox("Save validation overlays")
        form.addRow("", self.validation_check)
        return box

    def _build_window_group(self) -> QGroupBox:
        box = QGroupBox("Preview window")
        v = QVBoxLayout(box)
        self.meta_label = QLabel("—")
        self.meta_label.setObjectName("Hint")
        v.addWidget(self.meta_label)
        self.whole_check = QCheckBox("Preview whole file")
        v.addWidget(self.whole_check)
        row = QHBoxLayout()
        row.addWidget(QLabel("Start"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0.0, 10_000_000.0)
        self.start_spin.setDecimals(2)
        self.start_spin.setSuffix(" s")
        row.addWidget(self.start_spin, stretch=1)
        row.addWidget(QLabel("Length"))
        self.len_spin = QDoubleSpinBox()
        self.len_spin.setRange(0.25, MAX_PREVIEW_SECONDS)
        self.len_spin.setDecimals(2)
        self.len_spin.setValue(6.0)
        self.len_spin.setSuffix(" s")
        row.addWidget(self.len_spin, stretch=1)
        v.addLayout(row)
        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setRange(0, 0)
        v.addWidget(self.scrub)
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀ prev")
        self.btn_next = QPushButton("next ▶")
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        v.addLayout(nav)
        return box

    def _build_display_group(self) -> QGroupBox:
        box = QGroupBox("Display")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(spectro.COLORMAPS)
        form.addRow("Colormap", self.cmap_combo)
        self.dyn_spin = QDoubleSpinBox()
        self.dyn_spin.setRange(10.0, 160.0)
        self.dyn_spin.setValue(55.0)
        self.dyn_spin.setSingleStep(5.0)
        self.dyn_spin.setSuffix(" dB")
        form.addRow("Dynamic range", self.dyn_spin)
        self.denoise_check = QCheckBox("Denoise (subtract background)")
        self.denoise_check.setChecked(True)
        form.addRow("", self.denoise_check)
        self.wave_check = QCheckBox("Show waveform panel")
        self.wave_check.setChecked(True)
        form.addRow("", self.wave_check)
        self.btn_save_fig = QPushButton("Save current figure…")
        form.addRow("", self.btn_save_fig)
        return box

    # ---- right column ----
    def _build_right_column(self) -> QWidget:
        right = QSplitter(Qt.Vertical)
        self.tabs = QTabWidget()
        self.view = SpectrogramView()
        self.gallery = GalleryView()
        self.gallery.card_clicked.connect(self._on_card_clicked)
        self.tabs.addTab(self.view, "  Spectrogram  ")
        self.tabs.addTab(self._build_gallery_tab(), "  Gallery  ")
        right.addWidget(self.tabs)
        right.addWidget(self._build_detect_group())
        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 0)
        right.setSizes([560, 320])
        return right

    def _build_gallery_tab(self) -> QWidget:
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(8, 8, 8, 4)
        self.btn_gallery = QPushButton("Build gallery")
        self.btn_gallery.setObjectName("Primary")
        bar.addWidget(self.btn_gallery)
        bar.addWidget(QLabel("thumb window"))
        self.thumb_sec_spin = QDoubleSpinBox()
        self.thumb_sec_spin.setRange(0.5, 30.0)
        self.thumb_sec_spin.setValue(4.0)
        self.thumb_sec_spin.setSuffix(" s")
        bar.addWidget(self.thumb_sec_spin)
        self.gallery_progress = QProgressBar()
        self.gallery_progress.setTextVisible(False)
        self.gallery_progress.setMaximumHeight(8)
        bar.addWidget(self.gallery_progress, stretch=1)
        v.addLayout(bar)
        v.addWidget(self.gallery, stretch=1)
        return wrap

    def _build_detect_group(self) -> QGroupBox:
        box = QGroupBox("Detection")
        v = QVBoxLayout(box)

        ctl = QHBoxLayout()
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["Current window", "Whole file"])
        ctl.addWidget(QLabel("Scope"))
        ctl.addWidget(self.scope_combo)
        self.btn_detect = QPushButton("Detect vocalizations")
        self.btn_detect.setObjectName("Primary")
        self.btn_detect_all = QPushButton("Detect all → CSV")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("Danger")
        self.btn_cancel.setEnabled(False)
        ctl.addWidget(self.btn_detect)
        ctl.addWidget(self.btn_detect_all)
        ctl.addWidget(self.btn_cancel)
        self.btn_open_out = QPushButton("Open output ▸")
        self.btn_open_out.setEnabled(False)
        ctl.addWidget(self.btn_open_out)
        ctl.addStretch(1)
        v.addLayout(ctl)

        prog = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.phase_label = QLabel("")
        self.phase_label.setObjectName("Dim")
        prog.addWidget(self.progress, stretch=1)
        prog.addWidget(self.phase_label)
        v.addLayout(prog)

        split = QSplitter(Qt.Horizontal)
        self.table = DetectionsTable()
        split.addWidget(self.table)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(4000)
        self.log_view.setMinimumWidth(280)
        split.addWidget(self.log_view)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([640, 380])
        v.addWidget(split, stretch=1)
        return box

    def _build_menu(self) -> None:
        bar = self.menuBar()
        fm = bar.addMenu("&File")
        a1 = QAction("Add files…", self); a1.triggered.connect(self.add_files); fm.addAction(a1)
        a2 = QAction("Add folder…", self); a2.triggered.connect(self.add_folder); fm.addAction(a2)
        fm.addSeparator()
        a3 = QAction("Save current figure…", self); a3.triggered.connect(self.save_figure); fm.addAction(a3)
        fm.addSeparator()
        aq = QAction("Quit", self); aq.triggered.connect(self.close); fm.addAction(aq)
        hm = bar.addMenu("&Help")
        ab = QAction("About", self); ab.triggered.connect(self.show_about); hm.addAction(ab)

    # ------------------------------------------------------------- signals
    def _connect(self) -> None:
        self.btn_add_files.clicked.connect(self.add_files)
        self.btn_add_folder.clicked.connect(self.add_folder)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.clear_files)
        self.file_list.currentItemChanged.connect(self._on_selection_changed)

        self.species_combo.currentIndexChanged.connect(self._on_species_changed)
        self.auto_band_check.toggled.connect(self._on_band_toggled)

        self.whole_check.toggled.connect(self._on_whole_toggled)
        self.start_spin.editingFinished.connect(self._on_window_edited)
        self.len_spin.editingFinished.connect(self._on_length_edited)
        self.scrub.valueChanged.connect(self._on_scrub_moved)
        self.scrub.sliderReleased.connect(lambda: self.request_preview(self.current_path))
        self.btn_prev.clicked.connect(lambda: self._step_window(-1))
        self.btn_next.clicked.connect(lambda: self._step_window(+1))

        self.cmap_combo.currentIndexChanged.connect(self._rerender)
        self.dyn_spin.valueChanged.connect(self._rerender)
        self.denoise_check.toggled.connect(self._rerender)
        self.wave_check.toggled.connect(self._rerender)
        self.btn_save_fig.clicked.connect(self.save_figure)

        self.btn_detect.clicked.connect(self.detect_current)
        self.btn_detect_all.clicked.connect(self.detect_all)
        self.btn_cancel.clicked.connect(self.cancel_batch)
        self.btn_open_out.clicked.connect(self.open_output)
        self.table.row_activated.connect(self._on_table_row)
        self.btn_gallery.clicked.connect(self.build_gallery)

    # ----------------------------------------------------------- settings
    def display_settings(self) -> DisplaySettings:
        return DisplaySettings(
            colormap=self.cmap_combo.currentText(),
            dynamic_range_db=self.dyn_spin.value(),
            denoise=self.denoise_check.isChecked(),
            show_waveform=self.wave_check.isChecked(),
        )

    def detect_params(self) -> DetectParams:
        auto = self.auto_band_check.isChecked()
        return DetectParams(
            animal=self.species_combo.currentText(),
            bin_size=self.bin_spin.value(),
            lower_frequency_cutoff="default" if auto else int(self.flo_spin.value() * 1000),
            higher_frequency_cutoff="default" if auto else int(self.fhi_spin.value() * 1000),
            threads=self.threads_spin.value(),
            segmenter=self.segmenter_check.isChecked(),
            validation=self.validation_check.isChecked(),
        )

    def _preview_band(self) -> tuple[int, int]:
        p = self.detect_params()
        return engine.resolved_cutoffs(p.animal, p.lower_frequency_cutoff,
                                       p.higher_frequency_cutoff)

    def _on_species_changed(self) -> None:
        lo, hi = engine.DEFAULT_CUTOFFS.get(self.species_combo.currentText(), (0, 125000))
        self.flo_spin.blockSignals(True); self.fhi_spin.blockSignals(True)
        self.flo_spin.setValue(lo / 1000.0)
        self.fhi_spin.setValue(hi / 1000.0)
        self.flo_spin.blockSignals(False); self.fhi_spin.blockSignals(False)
        self._on_band_toggled()
        if self.current_path is not None:
            self.request_preview(self.current_path)

    def _on_band_toggled(self) -> None:
        auto = self.auto_band_check.isChecked()
        self.flo_spin.setEnabled(not auto)
        self.fhi_spin.setEnabled(not auto)

    # --------------------------------------------------------- file ops
    def add_files(self) -> None:
        patterns = "Audio (" + " ".join(f"*{e}" for e in spectro.AUDIO_EXTENSIONS) + ");;All files (*)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Add audio files", "", patterns)
        self._append_paths([pathlib.Path(p) for p in paths])

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add folder of audio")
        if not folder:
            return
        found = spectro.find_audio_files(folder, recursive=self.recursive_check.isChecked())
        if not found:
            self.log(f"No audio files found in {folder}", "warn")
            return
        before = self.file_list.count()
        self._append_paths(found)
        self.log(f"Added {self.file_list.count() - before} file(s) from {folder}.", "accent")

    def _append_paths(self, paths) -> None:
        existing = {self.file_list.item(i).data(ROLE_PATH) for i in range(self.file_list.count())}
        added, first_new = 0, None
        for p in paths:
            sp = str(p)
            if sp in existing:
                continue
            item = QListWidgetItem(p.name)
            item.setData(ROLE_PATH, sp)
            item.setToolTip(sp)
            self.file_list.addItem(item)
            existing.add(sp)
            added += 1
            first_new = first_new or item
        self._refresh_list_labels()
        if added and self.current_path is None and first_new is not None:
            self.file_list.setCurrentItem(first_new)
        self._update_counts()

    def _refresh_list_labels(self) -> None:
        paths = [self.file_list.item(i).data(ROLE_PATH) for i in range(self.file_list.count())]
        if not paths:
            return
        try:
            root = os.path.commonpath(paths) if len(paths) > 1 else os.path.dirname(paths[0])
        except ValueError:
            root = ""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            sp = item.data(ROLE_PATH)
            try:
                item.setText(os.path.relpath(sp, root) if root else os.path.basename(sp))
            except ValueError:
                item.setText(os.path.basename(sp))

    def remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
        self._refresh_list_labels()
        self._update_counts()

    def clear_files(self) -> None:
        self.file_list.clear()
        self.current_path = None
        self.current_info = None
        self._cur = None
        self.meta_label.setText("—")
        self._configure_window_controls()
        self.view.show_placeholder("Add audio files, select one to preview, then Detect vocalizations")
        self.table.clear_detections()
        for chip in (self.chip_file, self.chip_rate, self.chip_dur):
            chip.value.setText("—")
        self.chip_file.value.setText("no file")
        self._update_counts()

    def _update_counts(self) -> None:
        n = self.file_list.count()
        self.count_label.setText(f"{n} file{'s' if n != 1 else ''}")
        self.btn_detect_all.setText(f"Detect all → CSV ({n})" if n else "Detect all → CSV")
        if hasattr(self, "btn_gallery"):
            self.btn_gallery.setText(f"Build gallery ({n})" if n else "Build gallery")

    def all_paths(self):
        return [pathlib.Path(self.file_list.item(i).data(ROLE_PATH))
                for i in range(self.file_list.count())]

    # --------------------------------------------------------- preview
    def _on_selection_changed(self, current, _prev=None) -> None:
        if current is None:
            return
        self.current_path = pathlib.Path(current.data(ROLE_PATH))
        self.window_offset = 0.0
        self._load_info(self.current_path)
        self._configure_window_controls()
        self.request_preview(self.current_path)

    def _load_info(self, path) -> None:
        try:
            info = spectro.sound_info(path)
        except Exception as exc:
            info = None
            self.log(f"Could not read info for {path.name}: {exc}", "warn")
        self.current_info = info
        if info is not None:
            self.meta_label.setText(
                f"{info.duration:.1f} s  ·  {info.samplerate / 1000:.1f} kHz  ·  "
                f"{info.channels} ch  ·  Nyquist {info.nyquist / 1000:.0f} kHz")
            stem = path.stem
            self.chip_file.value.setText(stem if len(stem) <= 24 else stem[:23] + "…")
            self.chip_file.value.setToolTip(str(path))
            self.chip_rate.value.setText(f"{info.samplerate / 1000:.0f} kHz")
            mins = info.duration / 60.0
            self.chip_dur.value.setText(f"{info.duration:.0f} s" if info.duration < 120
                                        else f"{mins:.1f} min")
        else:
            self.meta_label.setText("—")

    def _configure_window_controls(self) -> None:
        info = self.current_info
        has = info is not None
        whole = self.whole_check.isChecked()
        length = self.len_spin.value()
        dur = info.duration if has else 0.0
        max_start = max(0.0, dur - length)
        self.window_offset = min(self.window_offset, max_start)
        self.start_spin.blockSignals(True); self.scrub.blockSignals(True)
        self.start_spin.setMaximum(max(0.0, max_start))
        self.start_spin.setValue(self.window_offset)
        self.scrub.setRange(0, int(round(max_start)))
        self.scrub.setValue(int(round(self.window_offset)))
        self.start_spin.blockSignals(False); self.scrub.blockSignals(False)
        windowed = has and not whole and max_start > 0
        for w in (self.start_spin, self.scrub, self.btn_prev, self.btn_next):
            w.setEnabled(windowed)
        self.len_spin.setEnabled(has and not whole)

    def _current_window(self):
        return self.window_offset, self.len_spin.value(), self.whole_check.isChecked()

    def _on_whole_toggled(self, _c) -> None:
        self._configure_window_controls()
        if self.current_path is not None:
            self.request_preview(self.current_path)

    def _on_length_edited(self) -> None:
        self._configure_window_controls()
        if self.current_path is not None:
            self.request_preview(self.current_path)

    def _on_window_edited(self) -> None:
        self.window_offset = self.start_spin.value()
        self.scrub.blockSignals(True); self.scrub.setValue(int(round(self.window_offset)))
        self.scrub.blockSignals(False)
        if self.current_path is not None:
            self.request_preview(self.current_path)

    def _on_scrub_moved(self, value: int) -> None:
        self.window_offset = float(value)
        self.start_spin.blockSignals(True); self.start_spin.setValue(self.window_offset)
        self.start_spin.blockSignals(False)
        if self.scrub.isSliderDown():
            length = self.len_spin.value()
            self.statusBar().showMessage(
                f"Window {self.window_offset:.1f}–{self.window_offset + length:.1f} s (release to compute)")
        elif self.current_path is not None:
            self.request_preview(self.current_path)

    def _step_window(self, direction: int) -> None:
        length = self.len_spin.value()
        offset = max(0.0, self.window_offset + direction * length)
        if self.current_info is not None:
            offset = min(offset, max(0.0, self.current_info.duration - length))
        self.window_offset = offset
        self.start_spin.blockSignals(True); self.scrub.blockSignals(True)
        self.start_spin.setValue(offset)
        self.scrub.setValue(int(round(offset)))
        self.start_spin.blockSignals(False); self.scrub.blockSignals(False)
        if self.current_path is not None:
            self.request_preview(self.current_path)

    def _detections_in_window(self):
        if self.current_path is None:
            return []
        dets = self._detections_by_path.get(str(self.current_path), [])
        if self.whole_check.isChecked():
            return dets
        o, length, _ = self._current_window()
        return [d for d in dets if d.end >= o and d.start <= o + length]

    def _rerender(self) -> None:
        if self._cur is None:
            return
        c = self._cur
        self.view.render(c["f"], c["t"], c["S"], c["samples"], c["sr"],
                         self.display_settings(), self._detections_in_window(),
                         title=self.current_path.name if self.current_path else None)

    def request_preview(self, path) -> None:
        if path is None:
            return
        self.current_path = pathlib.Path(path)
        self._pending_path = self.current_path
        self._maybe_start_preview()

    def _maybe_start_preview(self) -> None:
        if self._preview_worker is not None and self._preview_worker.isRunning():
            return
        if self._pending_path is None:
            return
        path = self._pending_path
        self._pending_path = None
        self.current_path = path
        start, length, whole = self._current_window()
        lo, hi = self._preview_band()
        self._preview_gen += 1
        gen = self._preview_gen
        win_txt = "whole file" if whole else f"{start:.1f}–{start + length:.1f} s"
        self.view.show_placeholder(f"Computing spectrogram …\n{path.name}  ({win_txt})")
        self.statusBar().showMessage(f"Computing {path.name} ({win_txt}) …")
        worker = PreviewWorker(path, gen, start, length, whole,
                               self.species_combo.currentText(), lo, hi)
        worker.done.connect(self._on_preview_done)
        worker.failed.connect(self._on_preview_failed)
        worker.finished.connect(lambda w=worker: self._on_preview_finished(w))
        self._preview_worker = worker
        self._live_workers.append(worker)
        worker.start()

    def _on_preview_finished(self, worker) -> None:
        self._reap(worker)
        if self._preview_worker is worker:
            self._preview_worker = None
        self._maybe_start_preview()

    def _on_preview_done(self, gen, f, t, S, samples, sr, stats) -> None:
        if gen != self._preview_gen:
            return
        self._cur = {"f": f, "t": t, "S": S, "samples": samples, "sr": sr,
                     "offset": stats["offset"]}
        self._rerender()
        self.chip_rate.value.setText(f"{sr / 1000:.0f} kHz")
        self.statusBar().showMessage(
            f"{self.current_path.name}  ·  {stats['duration']:.2f} s shown  ·  "
            f"spect {stats['shape'][0]}×{stats['shape'][1]}")

    def _on_preview_failed(self, gen, msg) -> None:
        if gen != self._preview_gen:
            return
        self._cur = None
        self.view.show_placeholder(f"Could not compute spectrogram:\n{msg}")
        self.log(f"Preview failed for {self.current_path}: {msg}", "error")

    # --------------------------------------------------------- detection
    def detect_current(self) -> None:
        if self.current_path is None:
            self.log("Select a file first.", "warn")
            return
        whole_scope = self.scope_combo.currentText() == "Whole file"
        window = None
        if not whole_scope:
            if self.whole_check.isChecked():
                window = None
            else:
                start, length, _ = self._current_window()
                window = (start, length)
        params = self.detect_params()
        scope_txt = "whole file" if window is None else f"window {window[0]:.1f}–{window[0] + window[1]:.1f} s"
        self.log(f"▶ Detecting ({params.animal}) on {self.current_path.name} — {scope_txt}"
                 + (" + SqueakOut" if params.segmenter else ""), "accent")
        self._set_busy(True)
        self.progress.setRange(0, 0)   # indeterminate
        worker = DetectWorker(self.current_path, params, window=window)
        worker.phase.connect(self._on_detect_phase)
        worker.log.connect(lambda m: self.log("   " + m))
        worker.done.connect(self._on_detect_done)
        worker.failed.connect(self._on_detect_failed)
        worker.finished.connect(lambda w=worker: self._on_detect_finished(w))
        self._detect_worker = worker
        self._live_workers.append(worker)
        worker.start()

    def _on_detect_phase(self, msg: str) -> None:
        self.phase_label.setText(msg)
        self.statusBar().showMessage(msg)

    def _on_detect_done(self, result) -> None:
        self._detections_by_path[str(result.path)] = result.detections
        self._rerender()
        self.table.set_detections(result.detections)
        summary = engine.summarize(result.detections, result.duration)
        self.view.metrics.set_detection_stats(summary)
        self._last_output_dir = result.output_dir
        self.btn_open_out.setEnabled(True)
        n = len(result.detections)
        self.log(f"■ {n} vocalization(s) detected · top class: {summary['dominant_class']} "
                 f"· {summary['per_min']:.1f}/min", "ok")
        if result.csv_path:
            self.log(f"   CSV → {result.csv_path}", "accent")
        self.tabs.setCurrentIndex(0)

    def _on_detect_failed(self, msg: str) -> None:
        self.log(f"Detection failed: {msg}", "error")
        self.statusBar().showMessage("Detection failed")

    def _on_detect_finished(self, worker) -> None:
        self._reap(worker)
        if self._detect_worker is worker:
            self._detect_worker = None
        self._set_busy(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.phase_label.setText("")

    def detect_all(self) -> None:
        paths = self.all_paths()
        if not paths:
            self.log("Add some audio files first.", "warn")
            return
        params = self.detect_params()
        self.log(f"▶ Batch detect: {len(paths)} file(s) · {params.animal} · whole files"
                 + (" + SqueakOut" if params.segmenter else "")
                 + " — writing <name>_outputs/<name>_stats.csv", "accent")
        self._set_busy(True, batch=True)
        self.progress.setRange(0, len(paths))
        self.progress.setValue(0)
        worker = BatchDetectWorker(paths, params)
        worker.progress.connect(self._on_batch_progress)
        worker.phase.connect(self._on_detect_phase)
        worker.file_ok.connect(self._on_batch_file_ok)
        worker.file_failed.connect(lambda name, msg: self.log(f"  ✗ {name} — {msg}", "error"))
        worker.finished_all.connect(self._on_batch_finished)
        worker.finished.connect(lambda w=worker: self._reap(w))
        self._batch_worker = worker
        self._live_workers.append(worker)
        worker.start()

    def _on_batch_progress(self, i, total, name) -> None:
        self.progress.setValue(i)
        self.statusBar().showMessage(f"[{i}/{total}] {name}")

    def _on_batch_file_ok(self, name, n_calls, csv) -> None:
        self.log(f"  ✓ {name} — {n_calls} calls → {csv}", "ok")

    def _on_batch_finished(self, n_ok, n_fail, cancelled) -> None:
        self._set_busy(False, batch=True)
        self._batch_worker = None
        tag = "cancelled" if cancelled else "complete"
        self.log(f"■ Batch {tag}: {n_ok} ok, {n_fail} failed.", "accent")
        self.statusBar().showMessage(f"Batch {tag}: {n_ok} ok, {n_fail} failed")
        self.phase_label.setText("")

    def cancel_batch(self) -> None:
        if self._batch_worker is not None:
            self._batch_worker.cancel()
            self.log("Cancellation requested (finishes current file) …", "warn")
            self.btn_cancel.setEnabled(False)

    def _on_table_row(self, start: float) -> None:
        if self.current_info is None:
            return
        length = self.len_spin.value()
        offset = max(0.0, min(start - length / 2.0, max(0.0, self.current_info.duration - length)))
        if self.whole_check.isChecked():
            self._rerender()
            return
        self.window_offset = offset
        self._configure_window_controls()
        self.request_preview(self.current_path)

    def _set_busy(self, busy: bool, batch: bool = False) -> None:
        self.btn_detect.setEnabled(not busy)
        self.btn_detect_all.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy and batch)
        for w in (self.species_combo, self.bin_spin, self.auto_band_check, self.flo_spin,
                  self.fhi_spin, self.threads_spin, self.segmenter_check, self.validation_check,
                  self.btn_add_files, self.btn_add_folder, self.btn_remove, self.btn_clear,
                  self.btn_gallery):
            w.setEnabled(not busy)
        self._on_band_toggled()

    def open_output(self) -> None:
        d = getattr(self, "_last_output_dir", None)
        if d and pathlib.Path(d).exists():
            try:
                subprocess.Popen(["xdg-open", str(d)])
            except Exception as exc:
                self.log(f"Could not open {d}: {exc}", "warn")

    # --------------------------------------------------------- gallery
    def build_gallery(self) -> None:
        paths = self.all_paths()
        if not paths:
            self.gallery.set_status("Add audio files first, then build the gallery.")
            self.tabs.setCurrentIndex(1)
            return
        if self._gallery_worker is not None and self._gallery_worker.isRunning():
            self._gallery_worker.cancel()
        labels = {self.file_list.item(i).data(ROLE_PATH): self.file_list.item(i).text()
                  for i in range(self.file_list.count())}
        self.gallery.clear()
        for p in paths:
            self.gallery.add_card(str(p), labels.get(str(p), p.name))
        self.gallery.set_status(f"Rendering {len(paths)} thumbnail(s) …")
        self.gallery_progress.setMaximum(len(paths))
        self.gallery_progress.setValue(0)
        self.btn_gallery.setEnabled(False)
        self.tabs.setCurrentIndex(1)
        worker = GalleryWorker(paths, self.display_settings(), self.thumb_sec_spin.value(),
                               self.species_combo.currentText())
        worker.thumb_ready.connect(self._on_thumb_ready)
        worker.thumb_failed.connect(self._on_thumb_failed)
        worker.progress.connect(self._on_gallery_progress)
        worker.finished_all.connect(self._on_gallery_finished)
        worker.finished.connect(lambda w=worker: self._reap(w))
        self._gallery_worker = worker
        self._live_workers.append(worker)
        worker.start()

    def _on_thumb_ready(self, idx, path, png, subtitle, offset) -> None:
        self._gallery_offsets[path] = offset
        card = self.gallery.card(path)
        if card is None:
            return
        pix = QPixmap(); pix.loadFromData(png, "PNG")
        card.set_image(pix)
        card.set_subtitle(subtitle)

    def _on_thumb_failed(self, idx, path, msg) -> None:
        card = self.gallery.card(path)
        if card is not None:
            card.set_failed(msg)

    def _on_gallery_progress(self, done, total) -> None:
        self.gallery_progress.setValue(done)
        self.gallery.set_status(f"Rendering thumbnails … {done}/{total}")

    def _on_gallery_finished(self, n_ok, n_fail, cancelled) -> None:
        self.btn_gallery.setEnabled(True)
        self._gallery_worker = None
        self.gallery.set_status(
            f"Gallery done: {n_ok} rendered, {n_fail} failed  ·  click a card to open it")

    def _on_card_clicked(self, path: str) -> None:
        offset = self._gallery_offsets.get(path, 0.0)
        for i in range(self.file_list.count()):
            if self.file_list.item(i).data(ROLE_PATH) == path:
                if self.file_list.currentRow() == i:
                    self._on_selection_changed(self.file_list.item(i))
                else:
                    self.file_list.setCurrentRow(i)
                break
        self.tabs.setCurrentIndex(0)
        if offset > 0 and self.current_info is not None and not self.whole_check.isChecked():
            self.window_offset = min(offset, max(0.0, self.current_info.duration - self.len_spin.value()))
            self._configure_window_controls()
            self.request_preview(self.current_path)

    # --------------------------------------------------------- misc
    def save_figure(self) -> None:
        if self._cur is None:
            self.log("Nothing to save yet — preview a file first.", "warn")
            return
        default = (self.current_path.stem if self.current_path else "spectrogram") + ".png"
        path, _ = QFileDialog.getSaveFileName(self, "Save figure", default,
                                              "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.view.save_figure(path, dpi=200)
            self.log(f"Saved figure → {path}")

    def show_about(self) -> None:
        QMessageBox.about(
            self, "About",
            "<h3>VocalPy USV Workbench</h3>"
            "<p>Detects, classifies, and segments animal ultrasonic vocalizations "
            "and exports a per-file CSV.</p>"
            "<p>Built on <b>gumadeiras/vocalpy</b> (inspired by VocalMat).</p>"
            "<p style='color:#7f9bad'>https://github.com/gumadeiras/vocalpy</p>")

    def log(self, message: str, level: str = "info") -> None:
        color = {"info": theme.TEXT_DIM, "accent": theme.ACCENT, "warn": theme.WARN,
                 "error": theme.ERROR, "ok": theme.OK}.get(level, theme.TEXT_DIM)
        self.log_view.appendHtml(f'<span style="color:{color}">{message}</span>')

    def _reap(self, worker) -> None:
        if worker in self._live_workers:
            self._live_workers.remove(worker)

    def _shutdown_workers(self) -> None:
        if self._batch_worker is not None:
            self._batch_worker.cancel()
        if self._gallery_worker is not None:
            self._gallery_worker.cancel()
        self._pending_path = None
        for w in list(self._live_workers):
            try:
                w.disconnect(self)
            except (RuntimeError, TypeError):
                pass
            if w.isRunning() and not w.wait(8000):
                w.terminate()
                w.wait()
        self._live_workers.clear()

    def closeEvent(self, event) -> None:
        self._shutdown_workers()
        event.accept()
        super().closeEvent(event)


def main() -> int:
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("VocalPy USV Workbench")
    app.setStyleSheet(theme.stylesheet())
    win = MainWindow()
    app.aboutToQuit.connect(win._shutdown_workers)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
