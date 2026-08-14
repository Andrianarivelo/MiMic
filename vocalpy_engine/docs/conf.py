import os
import sys
import tomllib

from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.pardir))

ROOT = Path(__file__).resolve().parent.parent
with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
    PYPROJECT = tomllib.load(pyproject_file)

project = "VocalPy"
copyright = "2020, Dietrich Lab"
author = "Gustavo Madeira Santana"
release = PYPROJECT["project"]["version"]
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
autodoc_mock_imports = [
    "cv2",
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "PIL",
    "scipy",
    "seaborn",
    "skimage",
    "soundfile",
    "torch",
    "torchvision",
    "yaml",
]
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_static_path = ["_static"]
html_extra_path = ["CNAME", ".nojekyll"]
html_theme_options = {
    "source_repository": "https://github.com/gumadeiras/vocalpy/",
    "source_branch": "main",
    "source_directory": "docs/",
}
