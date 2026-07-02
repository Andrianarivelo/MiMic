"""VocalPy Spectrogram Workbench.

A PySide6 desktop application for batch spectrogram processing and
interactive visualization built on top of the :mod:`vocalpy` library
(https://github.com/vocalpy/vocalpy).

The package is split into small, single-responsibility modules:

* :mod:`vpgui.theme`    - color palette and Qt stylesheet (dark workstation look)
* :mod:`vpgui.spectro`  - pure :mod:`vocalpy` logic (no Qt, no pyplot)
* :mod:`vpgui.plotting` - matplotlib rendering (Figure based, no Qt, no pyplot)
* :mod:`vpgui.workers`  - QThread workers for preview and batch processing
* :mod:`vpgui.widgets`  - reusable Qt widgets (canvas, panels)
* :mod:`vpgui.app`      - the main window and the ``main()`` entry point
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
