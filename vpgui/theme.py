"""Color palette and Qt stylesheet for the dark scientific-workstation look.

The aesthetic matches the rest of the lab's PySide6 tools: cyan / teal / blue
accents on a near-black ``#0b1017`` background.  The same color constants are
reused by :mod:`vpgui.plotting` so the matplotlib figures blend seamlessly into
the surrounding widgets.

This revision adds a "premium" pass: gradient header and primary buttons,
styled sliders, metric-tile / chip / badge classes, and a font stack that
prefers fonts actually present on the box (Ubuntu / Noto Sans) over the
previously-requested Inter / Segoe UI that were not installed.
"""

from __future__ import annotations

# --- core palette -----------------------------------------------------------
BG = "#0b1017"          # window background, near-black
BG_PANEL = "#0e1722"    # group boxes / side panels
BG_ELEV = "#13202d"     # elevated controls (inputs, list rows)
BG_ELEV2 = "#17293a"    # hover / raised
BG_PLOT = "#0c141d"     # matplotlib axes face
BG_HEADER_HI = "#122334"  # header gradient top
BG_HEADER_LO = "#0a1119"  # header gradient bottom

BORDER = "#1c2e3a"
BORDER_LIGHT = "#284559"
BORDER_GLOW = "#2b6b78"

TEXT = "#d4e6f1"        # primary text
TEXT_DIM = "#8fa9bb"    # secondary / labels
TEXT_FAINT = "#5d7788"  # disabled / hints

ACCENT = "#19c8d6"      # primary cyan
ACCENT_TEAL = "#2dd4bf"
ACCENT_BLUE = "#38bdf8"
ACCENT_PRESS = "#0fa6b3"
ACCENT_DEEP = "#0e7f8a"
ACCENT_INK = "#03161a"  # text on top of the accent fill

VIOLET = "#a78bfa"
PINK = "#f472b6"

OK = "#34d399"
WARN = "#f5a623"
ERROR = "#f87171"

# Font stack: prefer fonts actually present, fall back gracefully.
FONT_UI = '"Ubuntu", "Noto Sans", "DejaVu Sans", sans-serif'
FONT_MONO = '"Ubuntu Mono", "DejaVu Sans Mono", monospace'

# Default colormaps offered in the UI (perceptually uniform first).
COLORMAPS = [
    "magma", "inferno", "viridis", "plasma", "cividis",
    "turbo", "gray", "bone", "afmhot",
]


def stylesheet() -> str:
    """Return the global Qt stylesheet for the application."""
    return f"""
    QWidget {{
        background-color: {BG};
        color: {TEXT};
        font-family: {FONT_UI};
        font-size: 13px;
    }}
    QMainWindow, QDialog {{ background-color: {BG}; }}

    /* ---- header bar ---- */
    QFrame#HeaderBar {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {BG_HEADER_HI}, stop:1 {BG_HEADER_LO});
        border: 1px solid {BORDER};
        border-radius: 12px;
    }}
    QLabel#LogoGlyph {{ background: transparent; }}

    /* ---- group boxes / panels ---- */
    QGroupBox {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER};
        border-radius: 10px;
        margin-top: 12px;
        padding: 10px 10px 9px 10px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 11px;
        padding: 0 7px;
        color: {ACCENT};
        text-transform: uppercase;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
    }}

    QLabel {{ background: transparent; color: {TEXT}; }}
    QLabel#Dim {{ color: {TEXT_DIM}; }}
    QLabel#Hint {{ color: {TEXT_FAINT}; font-size: 11px; }}
    QLabel#Heading {{
        color: {TEXT}; font-size: 20px; font-weight: 800; letter-spacing: 0.5px;
    }}
    QLabel#Accent {{ color: {ACCENT}; font-weight: 700; }}
    QLabel#SubHeading {{ color: {TEXT_DIM}; font-size: 12px; letter-spacing: 1px; }}

    /* ---- metric chips (header, right side) ---- */
    QFrame#Chip {{
        background-color: {BG_ELEV};
        border: 1px solid {BORDER_LIGHT};
        border-radius: 9px;
    }}
    QLabel#ChipKey {{
        color: {TEXT_FAINT}; font-size: 9px; font-weight: 700;
        letter-spacing: 1.5px; text-transform: uppercase;
    }}
    QLabel#ChipVal {{ color: {ACCENT}; font-size: 15px; font-weight: 800; }}

    /* ---- metric tiles (under the spectrogram) ---- */
    QFrame#Tile {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER};
        border-radius: 9px;
    }}
    QFrame#Tile[accent="true"] {{ border-color: {BORDER_GLOW}; }}
    QLabel#TileKey {{
        color: {TEXT_FAINT}; font-size: 9px; font-weight: 700;
        letter-spacing: 1.4px; text-transform: uppercase;
    }}
    QLabel#TileVal {{ color: {TEXT}; font-size: 17px; font-weight: 800; }}
    QLabel#TileValAccent {{ color: {ACCENT}; font-size: 17px; font-weight: 800; }}
    QLabel#TileUnit {{ color: {TEXT_DIM}; font-size: 11px; }}

    /* ---- buttons ---- */
    QPushButton {{
        background-color: {BG_ELEV};
        border: 1px solid {BORDER_LIGHT};
        border-radius: 7px;
        padding: 7px 11px;
        color: {TEXT};
    }}
    QPushButton:hover {{
        border-color: {ACCENT}; color: {ACCENT}; background-color: {BG_ELEV2};
    }}
    QPushButton:pressed {{ background-color: {BORDER}; }}
    QPushButton:disabled {{ color: {TEXT_FAINT}; border-color: {BORDER}; }}

    QPushButton#Primary {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {ACCENT_TEAL}, stop:1 {ACCENT_DEEP});
        border: 1px solid {ACCENT};
        color: {ACCENT_INK};
        font-weight: 800;
        letter-spacing: 0.5px;
    }}
    QPushButton#Primary:hover {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {ACCENT_TEAL}, stop:1 {ACCENT});
        border-color: {ACCENT_TEAL};
    }}
    QPushButton#Primary:pressed {{ background: {ACCENT_PRESS}; }}
    QPushButton#Primary:disabled {{
        background: {BG_ELEV}; border-color: {BORDER}; color: {TEXT_FAINT};
    }}

    QPushButton#Preset {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1d3b52, stop:1 #142838);
        border: 1px solid {BORDER_GLOW};
        color: {ACCENT_BLUE};
        font-weight: 700;
    }}
    QPushButton#Preset:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}

    QPushButton#Danger:hover {{ border-color: {ERROR}; color: {ERROR}; }}

    /* ---- inputs ---- */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {BG_ELEV};
        border: 1px solid {BORDER_LIGHT};
        border-radius: 6px;
        padding: 5px 8px;
        selection-background-color: {ACCENT};
        selection-color: {ACCENT_INK};
    }}
    QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
        border-color: {BORDER_GLOW};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border-color: {ACCENT};
    }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
        background-color: {BG_PANEL};
        color: {TEXT_FAINT};
        border-color: {BORDER};
    }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox::down-arrow {{ width: 9px; height: 9px; }}
    QComboBox QAbstractItemView {{
        background-color: {BG_ELEV};
        border: 1px solid {ACCENT};
        border-radius: 6px;
        selection-background-color: {ACCENT};
        selection-color: {ACCENT_INK};
        outline: none;
        padding: 3px;
    }}
    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        background-color: {BG_ELEV2}; border: none; width: 16px;
    }}
    QSpinBox::up-button:hover, QSpinBox::down-button:hover,
    QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {BORDER_GLOW};
    }}

    /* ---- list ---- */
    QListWidget {{
        background-color: {BG_ELEV};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 3px;
        outline: none;
    }}
    QListWidget::item {{ padding: 6px 7px; border-radius: 5px; color: {TEXT}; }}
    QListWidget::item:hover {{ background-color: {BG_ELEV2}; }}
    QListWidget::item:selected {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ACCENT}, stop:1 {ACCENT_DEEP});
        color: {ACCENT_INK}; font-weight: 700;
    }}

    /* ---- text / log ---- */
    QPlainTextEdit, QTextEdit {{
        background-color: {BG_PLOT};
        border: 1px solid {BORDER};
        border-radius: 8px;
        font-family: {FONT_MONO};
        font-size: 12px;
        color: {TEXT_DIM};
        padding: 4px;
    }}

    /* ---- progress ---- */
    QProgressBar {{
        background-color: {BG_ELEV};
        border: 1px solid {BORDER};
        border-radius: 7px;
        text-align: center;
        color: {TEXT};
        height: 18px;
    }}
    QProgressBar::chunk {{
        border-radius: 6px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ACCENT_BLUE}, stop:1 {ACCENT_TEAL});
    }}

    /* ---- checkbox ---- */
    QCheckBox {{ spacing: 8px; background: transparent; }}
    QCheckBox::indicator {{
        width: 16px; height: 16px; border-radius: 5px;
        border: 1px solid {BORDER_LIGHT}; background: {BG_ELEV};
    }}
    QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
    QCheckBox::indicator:checked {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {ACCENT_TEAL}, stop:1 {ACCENT_DEEP});
        border-color: {ACCENT};
        image: none;
    }}

    /* ---- sliders (scrub) ---- */
    QSlider::groove:horizontal {{
        height: 5px; border-radius: 3px; background: {BG_ELEV};
    }}
    QSlider::sub-page:horizontal {{
        height: 5px; border-radius: 3px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ACCENT_DEEP}, stop:1 {ACCENT});
    }}
    QSlider::handle:horizontal {{
        width: 15px; height: 15px; margin: -6px 0; border-radius: 8px;
        background: {ACCENT}; border: 2px solid {BG};
    }}
    QSlider::handle:horizontal:hover {{ background: {ACCENT_TEAL}; }}
    QSlider::handle:horizontal:disabled {{ background: {BORDER_LIGHT}; }}

    /* ---- splitter / scrollbars / tooltips ---- */
    QSplitter::handle {{ background-color: {BG}; }}
    QSplitter::handle:horizontal {{ width: 6px; }}
    QSplitter::handle:vertical {{ height: 6px; }}
    QSplitter::handle:hover {{ background-color: {BORDER_GLOW}; }}

    QScrollBar:vertical {{ background: {BG}; width: 11px; margin: 0; }}
    QScrollBar::handle:vertical {{
        background: {BORDER_LIGHT}; border-radius: 5px; min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
    QScrollBar:horizontal {{ background: {BG}; height: 11px; margin: 0; }}
    QScrollBar::handle:horizontal {{
        background: {BORDER_LIGHT}; border-radius: 5px; min-width: 30px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: {ACCENT}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    QToolTip {{
        background-color: {BG_ELEV}; color: {TEXT};
        border: 1px solid {ACCENT}; border-radius: 5px; padding: 5px 7px;
    }}

    QStatusBar {{
        background-color: {BG_PANEL}; color: {TEXT_DIM};
        border-top: 1px solid {BORDER};
    }}
    QStatusBar::item {{ border: none; }}

    QMenuBar {{ background-color: {BG}; color: {TEXT}; }}
    QMenuBar::item {{ padding: 5px 10px; border-radius: 5px; }}
    QMenuBar::item:selected {{ background-color: {BG_ELEV}; color: {ACCENT}; }}
    QMenu {{ background-color: {BG_ELEV}; border: 1px solid {BORDER_LIGHT}; border-radius: 8px; padding: 4px; }}
    QMenu::item {{ padding: 6px 22px 6px 14px; border-radius: 5px; }}
    QMenu::item:selected {{ background-color: {ACCENT}; color: {ACCENT_INK}; }}

    QToolBar {{ background-color: {BG_PANEL}; border: none; spacing: 4px; }}

    /* ---- tabs ---- */
    QTabWidget::pane {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        top: -1px;
        background: {BG_PANEL};
    }}
    QTabBar::tab {{
        background: {BG_PANEL};
        color: {TEXT_DIM};
        border: 1px solid {BORDER};
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        padding: 8px 20px;
        margin-right: 3px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}
    QTabBar::tab:selected {{
        background: {BG_ELEV};
        color: {ACCENT};
        border-color: {ACCENT};
    }}
    QTabBar::tab:hover:!selected {{ color: {ACCENT_TEAL}; }}
    """
