#!/usr/bin/env python
"""No-install launcher for the VocalPy Spectrogram Workbench.

    conda activate vocalpy
    python run_vocalpy_gui.py
"""

import os
import sys

# Make sure the package next to this file is importable without installation.
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Expose the vendored gumadeiras/vocalpy engine when not installed editable.
LOCAL_VOCALPY = os.path.join(ROOT, "vocalpy_engine")
if os.path.isdir(os.path.join(LOCAL_VOCALPY, "vocalpy")):
    sys.path.insert(0, LOCAL_VOCALPY)

from vpgui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
