#!/usr/bin/env python
"""No-install launcher for the VocalPy Spectrogram Workbench.

    conda activate vocalpy
    python run_vocalpy_gui.py
"""

import os
import sys

# Make sure the package next to this file is importable without installation.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vpgui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
