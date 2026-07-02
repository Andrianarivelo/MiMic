"""VocalPy Spectrogram Workbench - main window and entry point.

A dark "scientific workstation" desktop app (PySide6) for:

* browsing / collecting audio files (or built-in VocalPy examples),
* interactively previewing spectrograms with three VocalPy backends
  (``librosa-db``, ``sat-multitaper``, ``soundsig-spectro``),
* batch-processing whole folders into ``.npz`` spectrograms and/or ``.png`` images.

Run with ``vocalpy-gui`` (after ``pip install -e .``) or ``python run_vocalpy_gui.py``.
"""

from __future__ import annotations

import pathlib
import sys

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

# Refuse to compute a preview whose spectrogram would exceed this many cells
# (~1 GB at 8 bytes). Protects against accidentally loading a whole 15-min,
# 384 kHz recording at a tiny hop length.
MAX_PREVIEW_CELLS = 1.3e8

import vocalpy as voc

from . import spectro, theme
from .gallery import GalleryView
from .spectro import SpectSettings
from .widgets import SpectrogramView
from .workers import BatchConfig, BatchWorker, GalleryWorker, PreviewWorker

ROLE_PATH = Qt.UserRole


class ExampleFetchWorker(QThread):
    """Download/locate a built-in VocalPy example clip off the UI thread."""

    done = Signal(str)    # path
    failed = Signal(str)  # message

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def run(self) -> None:
        try:
            path = voc.example(self.name, return_path=True)
            # vocalpy may return an ExampleData / path-like; normalize to str
            p = getattr(path, "path", path)
            self.done.emit(str(p))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VocalPy Spectrogram Workbench")
        self.resize(1340, 860)

        # --- state ---
        self.current_path: pathlib.Path | None = None
        self.current_sound: voc.Sound | None = None
        self.current_spect: voc.Spectrogram | None = None
        self.current_info: spectro.AudioInfo | None = None
        self.window_offset: float = 0.0   # absolute start (s) of the shown window
        self._render_offset: float = 0.0  # time offset used for the current render
        self._preview_gen = 0
        self._live_workers: list[QThread] = []
        self._preview_worker: PreviewWorker | None = None
        self._pending_path: pathlib.Path | None = None
        self._batch_worker: BatchWorker | None = None
        self._example_worker: ExampleFetchWorker | None = None
        self._gallery_worker: GalleryWorker | None = None

        self._build_ui()
        self._build_menu()
        self._connect()
        self._update_method_ui()
        self._configure_window_controls()
        self._update_counts()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 10, 12, 8)
        outer.setSpacing(8)

        outer.addLayout(self._build_header())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_column())
        splitter.addWidget(self._build_right_column())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([410, 1010])
        outer.addWidget(splitter, stretch=1)

        self.setStatusBar(self.statusBar())
        self.statusBar().showMessage(f"VocalPy {voc.__version__}  ·  ready")

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        left = QVBoxLayout()
        left.setSpacing(0)
        title = QLabel("VocalPy Spectrogram Workbench")
        title.setObjectName("Heading")
        sub = QLabel("batch processing · interactive spectrogram visualization")
        sub.setObjectName("SubHeading")
        left.addWidget(title)
        left.addWidget(sub)
        row.addLayout(left)
        row.addStretch(1)

        self.info_label = QLabel("no file loaded")
        self.info_label.setObjectName("Dim")
        self.info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.info_label)
        return row

    def _build_left_column(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 2, 0)
        v.setSpacing(7)
        v.addWidget(self._build_files_group())
        v.addWidget(self._build_window_group())
        v.addWidget(self._build_compute_group())
        v.addWidget(self._build_display_group())
        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(390)
        scroll.setMaximumWidth(470)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    def _build_files_group(self) -> QGroupBox:
        box = QGroupBox("Audio files")
        v = QVBoxLayout(box)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.setMinimumHeight(150)
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

        self.btn_example = QPushButton("Load built-in example ▾")
        v.addWidget(self.btn_example)

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
        self.len_spin.setRange(0.05, 3600.0)
        self.len_spin.setDecimals(2)
        self.len_spin.setValue(10.0)
        self.len_spin.setSuffix(" s")
        row.addWidget(self.len_spin, stretch=1)
        v.addLayout(row)

        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setRange(0, 0)
        self.scrub.setSingleStep(1)
        v.addWidget(self.scrub)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀ prev")
        self.btn_next = QPushButton("next ▶")
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        v.addLayout(nav)
        return box

    def _build_compute_group(self) -> QGroupBox:
        box = QGroupBox("Spectrogram method")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self.method_combo = QComboBox()
        self.method_combo.addItems(list(spectro.METHODS))
        form.addRow("Method", self.method_combo)

        self.nfft_spin = QSpinBox()
        self.nfft_spin.setRange(16, 16384)
        self.nfft_spin.setValue(512)
        self.nfft_spin.setSingleStep(64)
        form.addRow("n_fft", self.nfft_spin)

        self.hop_spin = QSpinBox()
        self.hop_spin.setRange(1, 8192)
        self.hop_spin.setValue(64)
        form.addRow("hop_length", self.hop_spin)

        # soundsig-specific parameters (shown only for that method)
        self.ss_rate_spin = QSpinBox()
        self.ss_rate_spin.setRange(100, 20000)
        self.ss_rate_spin.setValue(1000)
        self.ss_rate_spin.setSingleStep(100)
        self.ss_rate_row = self._add_row(form, "spec_sample_rate", self.ss_rate_spin)

        self.ss_spacing_spin = QSpinBox()
        self.ss_spacing_spin.setRange(1, 2000)
        self.ss_spacing_spin.setValue(50)
        self.ss_spacing_row = self._add_row(form, "freq_spacing (Hz)", self.ss_spacing_spin)

        self.ss_minf_spin = QSpinBox()
        self.ss_minf_spin.setRange(0, 200000)
        self.ss_minf_spin.setValue(0)
        self.ss_minf_row = self._add_row(form, "min_freq (Hz)", self.ss_minf_spin)

        self.ss_maxf_spin = QSpinBox()
        self.ss_maxf_spin.setRange(100, 400000)
        self.ss_maxf_spin.setValue(10000)
        self.ss_maxf_spin.setSingleStep(500)
        self.ss_maxf_row = self._add_row(form, "max_freq (Hz)", self.ss_maxf_spin)

        self.ss_nstd_spin = QSpinBox()
        self.ss_nstd_spin.setRange(1, 20)
        self.ss_nstd_spin.setValue(6)
        self.ss_nstd_row = self._add_row(form, "nstd", self.ss_nstd_spin)

        self.mono_check = QCheckBox("Mix down to mono before analysis")
        self.mono_check.setChecked(True)
        form.addRow("", self.mono_check)

        self.btn_recompute = QPushButton("Update preview")
        self.btn_recompute.setObjectName("Primary")
        form.addRow("", self.btn_recompute)
        return box

    def _build_display_group(self) -> QGroupBox:
        box = QGroupBox("Display")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(theme.COLORMAPS)
        form.addRow("Colormap", self.cmap_combo)

        self.dyn_spin = QDoubleSpinBox()
        self.dyn_spin.setRange(10.0, 160.0)
        self.dyn_spin.setValue(80.0)
        self.dyn_spin.setSingleStep(5.0)
        self.dyn_spin.setSuffix(" dB")
        form.addRow("Dynamic range", self.dyn_spin)

        self.wave_check = QCheckBox("Show waveform panel")
        self.wave_check.setChecked(True)
        form.addRow("", self.wave_check)

        self.flim_check = QCheckBox("Limit frequency axis")
        form.addRow("", self.flim_check)

        flim_row = QHBoxLayout()
        self.flim_min_spin = QDoubleSpinBox()
        self.flim_min_spin.setRange(0.0, 400.0)
        self.flim_min_spin.setValue(0.0)
        self.flim_min_spin.setSuffix(" kHz")
        self.flim_max_spin = QDoubleSpinBox()
        self.flim_max_spin.setRange(0.1, 400.0)
        self.flim_max_spin.setValue(15.0)
        self.flim_max_spin.setSuffix(" kHz")
        flim_row.addWidget(self.flim_min_spin)
        flim_row.addWidget(QLabel("→"))
        flim_row.addWidget(self.flim_max_spin)
        wrap = QWidget()
        wrap.setLayout(flim_row)
        flim_row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Range", wrap)

        self.btn_save_fig = QPushButton("Save current figure…")
        form.addRow("", self.btn_save_fig)
        return box

    def _build_right_column(self) -> QWidget:
        right = QSplitter(Qt.Vertical)
        self.tabs = QTabWidget()
        self.view = SpectrogramView()
        self.gallery = GalleryView()
        self.gallery.card_clicked.connect(self._on_card_clicked)
        self.tabs.addTab(self.view, "  Spectrogram  ")
        self.tabs.addTab(self._build_gallery_tab(), "  Gallery  ")
        right.addWidget(self.tabs)
        right.addWidget(self._build_batch_group())
        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 0)
        right.setSizes([600, 260])
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
        self.thumb_sec_spin.setRange(0.5, 120.0)
        self.thumb_sec_spin.setValue(8.0)
        self.thumb_sec_spin.setSuffix(" s")
        bar.addWidget(self.thumb_sec_spin)
        self.gallery_progress = QProgressBar()
        self.gallery_progress.setValue(0)
        self.gallery_progress.setTextVisible(False)
        self.gallery_progress.setMaximumHeight(8)
        bar.addWidget(self.gallery_progress, stretch=1)
        v.addLayout(bar)
        v.addWidget(self.gallery, stretch=1)
        return wrap

    def _build_batch_group(self) -> QGroupBox:
        box = QGroupBox("Batch processing")
        v = QVBoxLayout(box)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("choose an output folder for .npz / .png …")
        self.out_edit.setText(str(pathlib.Path.cwd() / "vocalpy_output"))
        self.btn_out = QPushButton("Browse…")
        out_row.addWidget(self.out_edit, stretch=1)
        out_row.addWidget(self.btn_out)
        v.addLayout(out_row)

        opt_row = QHBoxLayout()
        self.npz_check = QCheckBox("Save .npz spectrograms")
        self.npz_check.setChecked(True)
        self.png_check = QCheckBox("Save .png images")
        self.png_check.setChecked(True)
        opt_row.addWidget(self.npz_check)
        opt_row.addWidget(self.png_check)
        opt_row.addWidget(QLabel("PNG dpi"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(50, 600)
        self.dpi_spin.setValue(150)
        self.dpi_spin.setSingleStep(25)
        opt_row.addWidget(self.dpi_spin)
        opt_row.addStretch(1)
        opt_row.addWidget(QLabel("Max s/file"))
        self.maxsec_spin = QDoubleSpinBox()
        self.maxsec_spin.setRange(0.0, 100000.0)
        self.maxsec_spin.setDecimals(1)
        self.maxsec_spin.setValue(0.0)
        self.maxsec_spin.setSpecialValueText("all")
        self.maxsec_spin.setToolTip("Process only the first N seconds of each file (0 = whole file)")
        opt_row.addWidget(self.maxsec_spin)
        v.addLayout(opt_row)

        run_row = QHBoxLayout()
        self.btn_run = QPushButton("Process all files")
        self.btn_run.setObjectName("Primary")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("Danger")
        self.btn_cancel.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_cancel)
        run_row.addWidget(self.progress, stretch=1)
        v.addLayout(run_row)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)  # bound memory on long batches
        self.log_view.setMinimumHeight(90)
        self.log_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v.addWidget(self.log_view)
        return box

    @staticmethod
    def _add_row(form: QFormLayout, label: str, widget: QWidget) -> QWidget:
        """Add a row and return the label widget so the whole row can be hidden."""
        lbl = QLabel(label)
        form.addRow(lbl, widget)
        widget.setProperty("rowLabel", lbl)
        return lbl

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        act_add = QAction("Add files…", self)
        act_add.triggered.connect(self.add_files)
        act_folder = QAction("Add folder…", self)
        act_folder.triggered.connect(self.add_folder)
        file_menu.addAction(act_add)
        file_menu.addAction(act_folder)

        ex_menu = file_menu.addMenu("Load example")
        for name in spectro.EXAMPLE_CLIPS:
            a = QAction(name, self)
            a.triggered.connect(lambda _checked=False, n=name: self.load_example(n))
            ex_menu.addAction(a)

        file_menu.addSeparator()
        act_out = QAction("Set output folder…", self)
        act_out.triggered.connect(self.choose_output)
        file_menu.addAction(act_out)
        act_save = QAction("Save current figure…", self)
        act_save.triggered.connect(self.save_figure)
        file_menu.addAction(act_save)
        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        help_menu = bar.addMenu("&Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self.show_about)
        help_menu.addAction(act_about)

    # ------------------------------------------------------------- signals
    def _connect(self) -> None:
        self.btn_add_files.clicked.connect(self.add_files)
        self.btn_add_folder.clicked.connect(self.add_folder)
        self.btn_example.clicked.connect(self._show_example_menu)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.clear_files)
        self.file_list.currentItemChanged.connect(self._on_selection_changed)

        # preview window controls
        self.whole_check.toggled.connect(self._on_whole_toggled)
        self.start_spin.editingFinished.connect(self._on_window_edited)
        self.len_spin.editingFinished.connect(self._on_length_edited)
        self.scrub.valueChanged.connect(self._on_scrub_moved)
        self.scrub.sliderReleased.connect(self._on_scrub_released)
        self.btn_prev.clicked.connect(lambda: self._step_window(-1))
        self.btn_next.clicked.connect(lambda: self._step_window(+1))

        # compute params -> recompute
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        self.btn_recompute.clicked.connect(lambda: self.request_preview(self.current_path))
        for spin in (self.nfft_spin, self.hop_spin, self.ss_rate_spin,
                     self.ss_spacing_spin, self.ss_minf_spin, self.ss_maxf_spin,
                     self.ss_nstd_spin):
            spin.editingFinished.connect(self._on_compute_changed)
        self.mono_check.toggled.connect(self._on_compute_changed)

        # display params -> re-render only
        self.cmap_combo.currentIndexChanged.connect(self._on_display_changed)
        self.dyn_spin.valueChanged.connect(self._on_display_changed)
        self.wave_check.toggled.connect(self._on_display_changed)
        self.flim_check.toggled.connect(self._on_display_changed)
        self.flim_min_spin.valueChanged.connect(self._on_display_changed)
        self.flim_max_spin.valueChanged.connect(self._on_display_changed)
        self.btn_save_fig.clicked.connect(self.save_figure)

        # batch
        self.btn_out.clicked.connect(self.choose_output)
        self.btn_run.clicked.connect(self.start_batch)
        self.btn_cancel.clicked.connect(self.cancel_batch)

        # gallery
        self.btn_gallery.clicked.connect(self.build_gallery)

    # ----------------------------------------------------------- settings
    def read_settings(self) -> SpectSettings:
        return SpectSettings(
            method=self.method_combo.currentText(),
            n_fft=self.nfft_spin.value(),
            hop_length=self.hop_spin.value(),
            to_mono=self.mono_check.isChecked(),
            spec_sample_rate=self.ss_rate_spin.value(),
            freq_spacing=self.ss_spacing_spin.value(),
            min_freq=self.ss_minf_spin.value(),
            max_freq=self.ss_maxf_spin.value(),
            nstd=self.ss_nstd_spin.value(),
            colormap=self.cmap_combo.currentText(),
            dynamic_range_db=self.dyn_spin.value(),
            show_waveform=self.wave_check.isChecked(),
            limit_freq=self.flim_check.isChecked(),
            flim_min_khz=self.flim_min_spin.value(),
            flim_max_khz=self.flim_max_spin.value(),
        )

    def _update_method_ui(self) -> None:
        is_soundsig = self.method_combo.currentText() == "soundsig-spectro"
        # n_fft / hop_length are ignored by soundsig
        self.nfft_spin.setEnabled(not is_soundsig)
        self.hop_spin.setEnabled(not is_soundsig)
        ss_widgets = [
            self.ss_rate_spin, self.ss_spacing_spin, self.ss_minf_spin,
            self.ss_maxf_spin, self.ss_nstd_spin,
        ]
        for w in ss_widgets:
            w.setVisible(is_soundsig)
            lbl = w.property("rowLabel")
            if lbl is not None:
                lbl.setVisible(is_soundsig)

    # --------------------------------------------------------- file ops
    def add_files(self) -> None:
        patterns = "Audio (" + " ".join(f"*{e}" for e in spectro.AUDIO_EXTENSIONS) + ");;All files (*)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Add audio files", "", patterns)
        self._append_paths([pathlib.Path(p) for p in paths])

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add folder of audio")
        if not folder:
            return
        rec = self.recursive_check.isChecked()
        found = spectro.find_audio_files(folder, recursive=rec)
        if not found:
            self.log(f"No audio files found in {folder}", "warn")
            return
        before = self.file_list.count()
        self._append_paths(found)
        added = self.file_list.count() - before
        scope = "incl. sub-folders" if rec else "top level only"
        self.log(f"Added {added} audio file(s) from {folder} ({scope}).", "accent")
        self.log("  Tip: switch to the Gallery tab and Build gallery to preview them all.")

    def _append_paths(self, paths: list[pathlib.Path]) -> None:
        existing = {self.file_list.item(i).data(ROLE_PATH) for i in range(self.file_list.count())}
        added = 0
        first_new: QListWidgetItem | None = None
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
            if first_new is None:
                first_new = item
        self._refresh_list_labels()
        if added and self.current_path is None and first_new is not None:
            self.file_list.setCurrentItem(first_new)
        self._update_counts()

    def _refresh_list_labels(self) -> None:
        """Show each file relative to the common ancestor so nested files in a
        recursively-scanned folder are disambiguated (e.g. ``31096/6/clip.wav``)."""
        import os
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
                rel = os.path.relpath(sp, root) if root else os.path.basename(sp)
            except ValueError:
                rel = os.path.basename(sp)
            item.setText(rel)

    def remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
        self._refresh_list_labels()
        self._update_counts()

    def clear_files(self) -> None:
        self.file_list.clear()
        self.current_path = None
        self.current_sound = None
        self.current_spect = None
        self.current_info = None
        self.meta_label.setText("—")
        self._configure_window_controls()
        self.view.show_placeholder("Select an audio file to preview its spectrogram")
        self.info_label.setText("no file loaded")
        self._update_counts()

    def _update_counts(self) -> None:
        n = self.file_list.count()
        self.count_label.setText(f"{n} file{'s' if n != 1 else ''}")
        self.btn_run.setText(f"Process all files ({n})" if n else "Process all files")
        if hasattr(self, "btn_gallery"):
            self.btn_gallery.setText(f"Build gallery ({n})" if n else "Build gallery")

    def all_paths(self) -> list[pathlib.Path]:
        return [
            pathlib.Path(self.file_list.item(i).data(ROLE_PATH))
            for i in range(self.file_list.count())
        ]

    # --------------------------------------------------------- examples
    def _show_example_menu(self) -> None:
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        for name in spectro.EXAMPLE_CLIPS:
            menu.addAction(name, lambda n=name: self.load_example(n))
        menu.exec(self.btn_example.mapToGlobal(self.btn_example.rect().bottomLeft()))

    def load_example(self, name: str) -> None:
        self.statusBar().showMessage(f"Fetching example '{name}' …")
        self.btn_example.setEnabled(False)
        worker = ExampleFetchWorker(name)
        worker.done.connect(self._on_example_ready)
        worker.failed.connect(self._on_example_failed)
        worker.finished.connect(lambda w=worker: self._reap(w))
        self._example_worker = worker
        self._live_workers.append(worker)
        worker.start()

    def _on_example_ready(self, path: str) -> None:
        self.btn_example.setEnabled(True)
        self._append_paths([pathlib.Path(path)])
        # select the newly added example
        for i in range(self.file_list.count()):
            if self.file_list.item(i).data(ROLE_PATH) == path:
                self.file_list.setCurrentRow(i)
                break
        self.statusBar().showMessage(f"Loaded example: {pathlib.Path(path).name}")

    def _on_example_failed(self, msg: str) -> None:
        self.btn_example.setEnabled(True)
        self.log(f"Could not load example: {msg}", "error")
        self.statusBar().showMessage("Example download failed")

    # --------------------------------------------------------- preview
    def _on_selection_changed(self, current: QListWidgetItem | None, _prev=None) -> None:
        if current is None:
            return
        path = pathlib.Path(current.data(ROLE_PATH))
        self.current_path = path
        self.window_offset = 0.0
        self._load_info(path)
        self._configure_window_controls()
        self.request_preview(path)

    def _load_info(self, path: pathlib.Path) -> None:
        try:
            info = spectro.sound_info(path)
        except Exception as exc:
            info = None
            self.log(f"Could not read info for {path.name}: {exc}", "warn")
        self.current_info = info
        if info is not None:
            self.meta_label.setText(
                f"{info.duration:.1f} s  ·  {info.samplerate / 1000:.1f} kHz  ·  "
                f"{info.channels} ch  ·  Nyquist {info.nyquist / 1000:.0f} kHz"
            )
            # For very long files, hint that we are showing a window.
            if info.duration > self.len_spin.value() and not self.whole_check.isChecked():
                self.log(
                    f"{path.name}: {info.duration:.0f}s @ {info.samplerate/1000:.0f}kHz "
                    f"— previewing a {self.len_spin.value():.0f}s window (scrub to navigate).",
                    "accent",
                )
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

        self.start_spin.blockSignals(True)
        self.scrub.blockSignals(True)
        self.start_spin.setMaximum(max(0.0, max_start))
        self.start_spin.setValue(self.window_offset)
        self.scrub.setRange(0, int(round(max_start)))
        self.scrub.setValue(int(round(self.window_offset)))
        self.start_spin.blockSignals(False)
        self.scrub.blockSignals(False)

        windowed = has and not whole and max_start > 0
        for w in (self.start_spin, self.scrub, self.btn_prev, self.btn_next):
            w.setEnabled(windowed)
        self.len_spin.setEnabled(has and not whole)

    def _current_window(self) -> tuple[float, float, bool]:
        return self.window_offset, self.len_spin.value(), self.whole_check.isChecked()

    def _on_whole_toggled(self, _checked: bool) -> None:
        self._configure_window_controls()
        if self.current_path is not None:
            self.request_preview(self.current_path)

    def _on_length_edited(self) -> None:
        self._configure_window_controls()
        if self.current_path is not None:
            self.request_preview(self.current_path)

    def _on_window_edited(self) -> None:
        self.window_offset = self.start_spin.value()
        self.scrub.blockSignals(True)
        self.scrub.setValue(int(round(self.window_offset)))
        self.scrub.blockSignals(False)
        if self.current_path is not None:
            self.request_preview(self.current_path)

    def _on_scrub_moved(self, value: int) -> None:
        self.window_offset = float(value)
        self.start_spin.blockSignals(True)
        self.start_spin.setValue(self.window_offset)
        self.start_spin.blockSignals(False)
        length = self.len_spin.value()
        if self.scrub.isSliderDown():
            # dragging: just preview the target range, compute on release
            self.statusBar().showMessage(
                f"Window {self.window_offset:.1f}–{self.window_offset + length:.1f} s "
                f"(release to compute)"
            )
        elif self.current_path is not None:
            # keyboard / page-click: compute immediately
            self.request_preview(self.current_path)

    def _on_scrub_released(self) -> None:
        if self.current_path is not None:
            self.request_preview(self.current_path)

    def _step_window(self, direction: int) -> None:
        length = self.len_spin.value()
        offset = max(0.0, self.window_offset + direction * length)
        if self.current_info is not None:
            offset = min(offset, max(0.0, self.current_info.duration - length))
        self.window_offset = offset
        self.start_spin.blockSignals(True)
        self.scrub.blockSignals(True)
        self.start_spin.setValue(offset)
        self.scrub.setValue(int(round(offset)))
        self.start_spin.blockSignals(False)
        self.scrub.blockSignals(False)
        if self.current_path is not None:
            self.request_preview(self.current_path)

    def _on_method_changed(self) -> None:
        self._update_method_ui()
        self._on_compute_changed()

    def _on_compute_changed(self) -> None:
        if self.current_path is not None:
            self.request_preview(self.current_path)

    def _on_display_changed(self) -> None:
        # re-render cached spectrogram without recomputing
        if self.current_spect is not None:
            self.view.render(
                self.current_sound, self.current_spect, self.read_settings(),
                title=self.current_path.name if self.current_path else None,
                time_offset=self._render_offset,
            )

    def request_preview(self, path: pathlib.Path | None) -> None:
        """Coalesce preview requests: remember the latest, run one at a time.

        Rapid file selection / scrubbing therefore never piles up concurrent
        heavy compute threads -- only the most recent request actually runs.
        """
        if path is None:
            return
        self.current_path = pathlib.Path(path)
        self._pending_path = self.current_path
        self._maybe_start_preview()

    def _maybe_start_preview(self) -> None:
        if self._preview_worker is not None and self._preview_worker.isRunning():
            return  # a preview is in flight; the pending one runs when it finishes
        if self._pending_path is None:
            return
        path = self._pending_path
        self._pending_path = None
        self.current_path = path
        start, length, whole = self._current_window()

        # Memory guard: refuse previews whose spectrogram would be enormous.
        info = self.current_info
        if info is not None:
            n_samples = info.frames if whole else min(
                info.frames, int(length * info.samplerate)
            )
            cells = spectro.estimate_spectrogram_cells(
                info.samplerate, n_samples, self.read_settings()
            )
            if cells > MAX_PREVIEW_CELLS:
                gb = cells * 8 / 1e9
                self.current_spect = None
                self.view.show_placeholder(
                    f"Preview would be ~{gb:.1f} GB — too large to display.\n\n"
                    "Uncheck 'Preview whole file', shorten the window,\n"
                    "or increase hop_length / spec_sample_rate."
                )
                self.statusBar().showMessage("Preview skipped: window too large")
                return

        self._render_offset = 0.0 if whole else start
        self._preview_gen += 1
        gen = self._preview_gen
        win_txt = "whole file" if whole else f"{start:.1f}–{start + length:.1f} s"
        self.view.show_placeholder(
            f"Computing spectrogram …\n{path.name}  ({win_txt})"
        )
        self.statusBar().showMessage(f"Computing {path.name} ({win_txt}) …")
        worker = PreviewWorker(
            path, self.read_settings(), gen,
            start_sec=start, length_sec=length, whole=whole,
        )
        worker.done.connect(self._on_preview_done)
        worker.failed.connect(self._on_preview_failed)
        worker.finished.connect(lambda w=worker: self._on_preview_worker_finished(w))
        self._preview_worker = worker
        self._live_workers.append(worker)
        worker.start()

    def _on_preview_worker_finished(self, worker: QThread) -> None:
        self._reap(worker)
        if self._preview_worker is worker:
            self._preview_worker = None
        self._maybe_start_preview()  # run the most recent pending request, if any

    def _on_preview_done(self, gen: int, sound, spect, stats: dict) -> None:
        if gen != self._preview_gen:
            return  # a newer request superseded this one
        self.current_sound = sound
        self.current_spect = spect
        self.view.render(
            sound, spect, self.read_settings(),
            title=self.current_path.name if self.current_path else None,
            time_offset=self._render_offset,
        )
        shape = stats["shape"]
        win = ""
        if self.current_info is not None and not self.whole_check.isChecked() \
                and self.current_info.duration > sound.duration + 1e-3:
            win = (f"   ·   window {self._render_offset:.1f}–"
                   f"{self._render_offset + sound.duration:.1f} s")
        self.info_label.setText(
            f"{self.current_path.name}   ·   {sound.samplerate / 1000:.1f} kHz   ·   "
            f"{sound.channels} ch   ·   {sound.duration:.2f} s shown{win}   ·   "
            f"spect {shape[1]}×{shape[2]}"
        )
        self.statusBar().showMessage(
            f"Done: {self.current_path.name}  "
            f"(f {stats['f_min']:.0f}–{stats['f_max']:.0f} Hz, "
            f"{stats['vmin']:.1f}…{stats['vmax']:.1f} dB)"
        )

    def _on_preview_failed(self, gen: int, msg: str) -> None:
        if gen != self._preview_gen:
            return
        self.view.show_placeholder(f"Could not compute spectrogram:\n{msg}")
        self.info_label.setText("error")
        self.statusBar().showMessage("Preview failed")
        self.log(f"Preview failed for {self.current_path}: {msg}", "error")

    # --------------------------------------------------------- batch
    def choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", self.out_edit.text())
        if folder:
            self.out_edit.setText(folder)

    def start_batch(self) -> None:
        paths = self.all_paths()
        if not paths:
            self.log("Add some audio files first.", "warn")
            return
        if not (self.npz_check.isChecked() or self.png_check.isChecked()):
            self.log("Select at least one output format (.npz or .png).", "warn")
            return
        out_text = self.out_edit.text().strip()
        if not out_text:
            self.log("Set an output folder first (Browse…).", "warn")
            return
        out_dir = pathlib.Path(out_text)
        max_sec = self.maxsec_spin.value()
        config = BatchConfig(
            output_dir=out_dir,
            save_npz=self.npz_check.isChecked(),
            save_png=self.png_check.isChecked(),
            png_dpi=self.dpi_spin.value(),
            max_seconds=max_sec,
        )
        settings = self.read_settings()

        self.progress.setMaximum(len(paths))
        self.progress.setValue(0)
        self._set_busy(True)
        cap = f"first {max_sec:g}s/file" if max_sec > 0 else "whole files"
        self.log(
            f"▶ Batch start: {len(paths)} file(s) · method={settings.method} · "
            f"{cap} · out={out_dir}", "accent",
        )

        worker = BatchWorker(paths, settings, config)
        worker.progress.connect(self._on_batch_progress)
        worker.file_ok.connect(self._on_batch_file_ok)
        worker.file_failed.connect(self._on_batch_file_failed)
        worker.finished_all.connect(self._on_batch_finished)
        worker.finished.connect(lambda w=worker: self._reap(w))
        self._batch_worker = worker
        self._live_workers.append(worker)
        worker.start()

    def cancel_batch(self) -> None:
        if self._batch_worker is not None:
            self._batch_worker.cancel()
            self.log("Cancellation requested …", "warn")
            self.btn_cancel.setEnabled(False)

    def _on_batch_progress(self, i: int, total: int, name: str) -> None:
        self.progress.setValue(i)
        self.statusBar().showMessage(f"[{i}/{total}] {name}")

    def _on_batch_file_ok(self, name: str, outputs: str) -> None:
        self.log(f"  ✓ {name}  →  {outputs}")

    def _on_batch_file_failed(self, name: str, msg: str) -> None:
        self.log(f"  ✗ {name}  —  {msg}", "error")

    def _on_batch_finished(self, n_ok: int, n_fail: int, cancelled: bool) -> None:
        self._set_busy(False)
        self._batch_worker = None
        tag = "cancelled" if cancelled else "complete"
        self.log(f"■ Batch {tag}: {n_ok} ok, {n_fail} failed.", "accent")
        self.statusBar().showMessage(f"Batch {tag}: {n_ok} ok, {n_fail} failed")
        if not cancelled and n_ok and n_fail == 0:
            self.progress.setValue(self.progress.maximum())

    def _set_busy(self, busy: bool) -> None:
        """Lock controls that must not change while a batch is running."""
        self.btn_run.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        for w in (
            self.btn_add_files, self.btn_add_folder, self.btn_example,
            self.btn_remove, self.btn_clear, self.method_combo, self.nfft_spin,
            self.hop_spin, self.mono_check, self.btn_recompute, self.npz_check,
            self.png_check, self.dpi_spin, self.maxsec_spin, self.btn_out,
            self.ss_rate_spin, self.ss_spacing_spin, self.ss_minf_spin,
            self.ss_maxf_spin, self.ss_nstd_spin,
        ):
            w.setEnabled(not busy)
        if hasattr(self, "btn_gallery"):
            self.btn_gallery.setEnabled(not busy)

    # --------------------------------------------------------- gallery
    def build_gallery(self) -> None:
        paths = self.all_paths()
        if not paths:
            self.gallery.set_status("Add audio files first, then build the gallery.")
            self.tabs.setCurrentWidget(self._gallery_container())
            return
        if self._gallery_worker is not None and self._gallery_worker.isRunning():
            self._gallery_worker.cancel()

        labels = {
            self.file_list.item(i).data(ROLE_PATH): self.file_list.item(i).text()
            for i in range(self.file_list.count())
        }
        self.gallery.clear()
        for p in paths:
            self.gallery.add_card(str(p), labels.get(str(p), p.name))
        self.gallery.set_status(f"Rendering {len(paths)} thumbnail(s) …")
        self.gallery_progress.setMaximum(len(paths))
        self.gallery_progress.setValue(0)
        self.btn_gallery.setEnabled(False)
        self.tabs.setCurrentIndex(1)

        worker = GalleryWorker(paths, self.read_settings(), self.thumb_sec_spin.value())
        worker.thumb_ready.connect(self._on_thumb_ready)
        worker.thumb_failed.connect(self._on_thumb_failed)
        worker.progress.connect(self._on_gallery_progress)
        worker.finished_all.connect(self._on_gallery_finished)
        worker.finished.connect(lambda w=worker: self._reap(w))
        self._gallery_worker = worker
        self._live_workers.append(worker)
        worker.start()

    def _gallery_container(self) -> QWidget:
        return self.tabs.widget(1)

    def _on_thumb_ready(self, idx: int, path: str, png: bytes, subtitle: str) -> None:
        card = self.gallery.card(path)
        if card is None:
            return
        pix = QPixmap()
        pix.loadFromData(png, "PNG")
        card.set_image(pix)
        card.set_subtitle(subtitle)

    def _on_thumb_failed(self, idx: int, path: str, msg: str) -> None:
        card = self.gallery.card(path)
        if card is not None:
            card.set_failed(msg)

    def _on_gallery_progress(self, done: int, total: int) -> None:
        self.gallery_progress.setValue(done)
        self.gallery.set_status(f"Rendering thumbnails … {done}/{total}")

    def _on_gallery_finished(self, n_ok: int, n_fail: int, cancelled: bool) -> None:
        self.btn_gallery.setEnabled(True)
        self._gallery_worker = None
        tag = "cancelled" if cancelled else "done"
        self.gallery.set_status(
            f"Gallery {tag}: {n_ok} rendered, {n_fail} failed  ·  click a card to open it"
        )

    def _on_card_clicked(self, path: str) -> None:
        for i in range(self.file_list.count()):
            if self.file_list.item(i).data(ROLE_PATH) == path:
                self.file_list.setCurrentRow(i)
                break
        self.tabs.setCurrentIndex(0)  # show the interactive spectrogram

    # --------------------------------------------------------- misc
    def save_figure(self) -> None:
        if self.current_spect is None:
            self.log("Nothing to save yet — preview a file first.", "warn")
            return
        default = (self.current_path.stem if self.current_path else "spectrogram") + ".png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", default,
            "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if path:
            self.view.save_figure(path, dpi=200)
            self.log(f"Saved figure → {path}")
            self.statusBar().showMessage(f"Saved {path}")

    def show_about(self) -> None:
        QMessageBox.about(
            self, "About",
            "<h3>VocalPy Spectrogram Workbench</h3>"
            f"<p>Built on VocalPy {voc.__version__} — a core package for "
            "acoustic communication research.</p>"
            "<p>Preview and batch-export spectrograms with the librosa-db, "
            "sat-multitaper and soundsig-spectro backends.</p>"
            "<p style='color:#7f9bad'>https://github.com/vocalpy/vocalpy</p>",
        )

    def log(self, message: str, level: str = "info") -> None:
        color = {
            "info": theme.TEXT_DIM, "accent": theme.ACCENT,
            "warn": theme.WARN, "error": theme.ERROR, "ok": theme.OK,
        }.get(level, theme.TEXT_DIM)
        self.log_view.appendHtml(f'<span style="color:{color}">{message}</span>')

    def _reap(self, worker: QThread) -> None:
        """Drop a finished worker reference so it can be garbage collected."""
        if worker in self._live_workers:
            self._live_workers.remove(worker)

    def _shutdown_workers(self) -> None:
        """Deterministic shutdown: never let a running QThread be destroyed, and
        stop queued signals from firing into a half-torn-down window."""
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
            if w.isRunning() and not w.wait(5000):
                w.terminate()
                w.wait()
        self._live_workers.clear()

    def closeEvent(self, event) -> None:
        self._shutdown_workers()
        event.accept()
        super().closeEvent(event)


def main() -> int:
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("VocalPy Spectrogram Workbench")
    app.setStyleSheet(theme.stylesheet())
    win = MainWindow()
    # Safety net: also clean up workers on any quit path (not just window close).
    app.aboutToQuit.connect(win._shutdown_workers)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
