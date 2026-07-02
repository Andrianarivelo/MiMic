"""Color palette and Qt stylesheet for the dark scientific-workstation look.

The aesthetic matches the rest of the lab's PySide6 tools: cyan / teal / blue
accents on a near-black ``#0b1017`` background.  The same color constants are
reused by :mod:`vpgui.plotting` so the matplotlib figures blend seamlessly into
the surrounding widgets.
"""

from __future__ import annotations

# --- core palette -----------------------------------------------------------
BG = "#0b1017"          # window background, near-black
BG_PANEL = "#0e1722"    # group boxes / side panels
BG_ELEV = "#13202d"     # elevated controls (inputs, list rows)
BG_PLOT = "#0c141d"     # matplotlib axes face

BORDER = "#1c2e3a"
BORDER_LIGHT = "#284559"

TEXT = "#d4e6f1"        # primary text
TEXT_DIM = "#7f9bad"    # secondary / labels
TEXT_FAINT = "#56707f"  # disabled / hints

ACCENT = "#19c8d6"      # primary cyan
ACCENT_TEAL = "#2dd4bf"
ACCENT_BLUE = "#38bdf8"
ACCENT_PRESS = "#0fa6b3"

OK = "#34d399"
WARN = "#f5a623"
ERROR = "#f87171"

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
        font-family: "Inter", "Segoe UI", "DejaVu Sans", sans-serif;
        font-size: 13px;
    }}
    QMainWindow, QDialog {{ background-color: {BG}; }}

    /* ---- group boxes / panels ---- */
    QGroupBox {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER};
        border-radius: 8px;
        margin-top: 11px;
        padding: 8px 9px 8px 9px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 6px;
        color: {ACCENT};
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 1px;
    }}

    QLabel {{ background: transparent; color: {TEXT}; }}
    QLabel#Dim {{ color: {TEXT_DIM}; }}
    QLabel#Hint {{ color: {TEXT_FAINT}; font-size: 11px; }}
    QLabel#Heading {{
        color: {ACCENT}; font-size: 17px; font-weight: 700; letter-spacing: 1px;
    }}
    QLabel#SubHeading {{ color: {TEXT_DIM}; font-size: 12px; }}

    /* ---- buttons ---- */
    QPushButton {{
        background-color: {BG_ELEV};
        border: 1px solid {BORDER_LIGHT};
        border-radius: 6px;
        padding: 6px 10px;
        color: {TEXT};
    }}
    QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
    QPushButton:pressed {{ background-color: {BORDER}; }}
    QPushButton:disabled {{ color: {TEXT_FAINT}; border-color: {BORDER}; }}

    QPushButton#Primary {{
        background-color: {ACCENT};
        border: 1px solid {ACCENT};
        color: #03161a;
        font-weight: 700;
    }}
    QPushButton#Primary:hover {{ background-color: {ACCENT_TEAL}; border-color: {ACCENT_TEAL}; }}
    QPushButton#Primary:pressed {{ background-color: {ACCENT_PRESS}; }}
    QPushButton#Primary:disabled {{
        background-color: {BG_ELEV}; border-color: {BORDER}; color: {TEXT_FAINT};
    }}
    QPushButton#Danger:hover {{ border-color: {ERROR}; color: {ERROR}; }}

    /* ---- inputs ---- */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {BG_ELEV};
        border: 1px solid {BORDER_LIGHT};
        border-radius: 5px;
        padding: 4px 8px;
        selection-background-color: {ACCENT};
        selection-color: #03161a;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border-color: {ACCENT};
    }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
        background-color: {BG_PANEL};
        color: {TEXT_FAINT};
        border-color: {BORDER};
    }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{
        background-color: {BG_ELEV};
        border: 1px solid {ACCENT};
        selection-background-color: {ACCENT};
        selection-color: #03161a;
        outline: none;
    }}
    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        background-color: {BG_ELEV}; border: none; width: 16px;
    }}

    /* ---- list ---- */
    QListWidget {{
        background-color: {BG_ELEV};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 2px;
        outline: none;
    }}
    QListWidget::item {{ padding: 5px 6px; border-radius: 4px; }}
    QListWidget::item:hover {{ background-color: {BORDER}; }}
    QListWidget::item:selected {{
        background-color: {ACCENT}; color: #03161a;
    }}

    /* ---- text / log ---- */
    QPlainTextEdit, QTextEdit {{
        background-color: {BG_PLOT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
        font-size: 12px;
        color: {TEXT_DIM};
    }}

    /* ---- progress ---- */
    QProgressBar {{
        background-color: {BG_ELEV};
        border: 1px solid {BORDER};
        border-radius: 6px;
        text-align: center;
        color: {TEXT};
        height: 18px;
    }}
    QProgressBar::chunk {{
        border-radius: 5px;
        background-color: {ACCENT};
    }}

    /* ---- checkbox ---- */
    QCheckBox {{ spacing: 7px; background: transparent; }}
    QCheckBox::indicator {{
        width: 15px; height: 15px; border-radius: 4px;
        border: 1px solid {BORDER_LIGHT}; background: {BG_ELEV};
    }}
    QCheckBox::indicator:checked {{
        background: {ACCENT}; border-color: {ACCENT};
        image: none;
    }}

    /* ---- splitter / scrollbars / tooltips ---- */
    QSplitter::handle {{ background-color: {BORDER}; }}
    QSplitter::handle:horizontal {{ width: 3px; }}
    QSplitter::handle:vertical {{ height: 3px; }}

    QScrollBar:vertical {{ background: {BG}; width: 11px; margin: 0; }}
    QScrollBar::handle:vertical {{
        background: {BORDER_LIGHT}; border-radius: 5px; min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
    QScrollBar:horizontal {{ background: {BG}; height: 11px; margin: 0; }}
    QScrollBar::handle:horizontal {{
        background: {BORDER_LIGHT}; border-radius: 5px; min-width: 28px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: {ACCENT}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}

    QToolTip {{
        background-color: {BG_ELEV}; color: {TEXT};
        border: 1px solid {ACCENT}; border-radius: 4px; padding: 4px 6px;
    }}

    QStatusBar {{ background-color: {BG_PANEL}; color: {TEXT_DIM}; }}
    QStatusBar::item {{ border: none; }}

    QMenuBar {{ background-color: {BG_PANEL}; color: {TEXT}; }}
    QMenuBar::item:selected {{ background-color: {BORDER}; color: {ACCENT}; }}
    QMenu {{ background-color: {BG_ELEV}; border: 1px solid {BORDER_LIGHT}; }}
    QMenu::item:selected {{ background-color: {ACCENT}; color: #03161a; }}

    QToolBar {{ background-color: {BG_PANEL}; border: none; spacing: 4px; }}

    /* ---- tabs ---- */
    QTabWidget::pane {{
        border: 1px solid {BORDER};
        border-radius: 6px;
        top: -1px;
    }}
    QTabBar::tab {{
        background: {BG_PANEL};
        color: {TEXT_DIM};
        border: 1px solid {BORDER};
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        padding: 6px 14px;
        margin-right: 2px;
        font-weight: 600;
    }}
    QTabBar::tab:selected {{
        background: {BG_ELEV};
        color: {ACCENT};
        border-color: {ACCENT};
    }}
    QTabBar::tab:hover {{ color: {ACCENT}; }}
    """
