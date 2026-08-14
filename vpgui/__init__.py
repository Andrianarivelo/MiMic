"""VocalPy USV Workbench.

A PySide6 desktop application that detects, classifies and segments animal
ultrasonic vocalizations and exports a per-file CSV, built on the vendored
``gumadeiras/vocalpy`` engine (https://github.com/gumadeiras/vocalpy).

The package is split into small, single-responsibility modules:

* :mod:`vpgui.theme`    - color palette and Qt stylesheet (dark workstation look)
* :mod:`vpgui.engine`   - adapter over the vendored ``vocalpy`` detection pipeline
* :mod:`vpgui.spectro`  - audio discovery + engine-consistent display spectrograms
* :mod:`vpgui.cli`      - command-line entry point (detect files/folders -> CSV)
* :mod:`vpgui.plotting` - matplotlib rendering + detection overlays (no Qt/pyplot)
* :mod:`vpgui.workers`  - QThread workers (preview, detect, batch, gallery)
* :mod:`vpgui.widgets`  - metric tiles + interactive spectrogram canvas
* :mod:`vpgui.results`  - detections table
* :mod:`vpgui.app`      - the main window and the ``main()`` entry point
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
