# VocalPy
![Python Versions](https://img.shields.io/badge/python-3.12-blue)
![Platforms](https://img.shields.io/badge/platform-linux--64%20%7C%20osx--64-lightgrey)
![build](https://github.com/gumadeiras/vocalpy/workflows/build/badge.svg?branch=main)
[![CodeFactor](https://www.codefactor.io/repository/github/gumadeiras/vocalpy/badge?s=e1ba6c8796b9923a3cdcfd1e51fcf368a743ab83)](https://www.codefactor.io/repository/github/gumadeiras/vocalpy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
![License](https://img.shields.io/badge/license-Apache%202-blue)

VocalPy detects, classifies, and segments animal vocalizations from audio recordings. It is inspired by [VocalMat](https://github.com/ahof1704/VocalMat).

Point it at a `.wav` file and it runs a full analysis pipeline: it chunks the audio for parallel processing, detects vocalizations using a spectrogram-based method tuned for the selected species, removes noise candidates with a pretrained classifier, assigns vocalization-type labels, and optionally produces binary segmentation masks for each detected call using [SqueakOut](https://github.com/gumadeiras/squeakout). Results land in a CSV, per-vocal spectrogram images, and serialized recording objects — all in an output folder next to the audio file.

**Supported species:** mouse, rat, guinea pig

## Installation

Requires Python 3.12 and [Git LFS](https://git-lfs.github.com/) (bundled model checkpoints are stored via Git LFS — clone without it and the models will be missing).

With `micromamba` (recommended):

```sh
micromamba create -y -n vocalpy python=3.12
micromamba activate vocalpy
git clone https://github.com/gumadeiras/vocalpy.git
cd vocalpy
pip install --upgrade pip
pip install -r requirements-dev.txt
```

With `venv`:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
git clone https://github.com/gumadeiras/vocalpy.git
cd vocalpy
pip install --upgrade pip
pip install -r requirements-dev.txt
```

## Quick start

```sh
# Detect and classify mouse ultrasonic vocalizations (USVs)
vocalpy -p /path/to/recording.wav

# Use the rat pipeline instead
vocalpy -a rat -p /path/to/recording.wav

# Also run autoencoder-based segmentation — produces a binary mask per vocal saved under cnn_mask/
vocalpy -p /path/to/recording.wav --segmenter

# Save spectrogram-overlay images so you can manually verify detections
vocalpy -p /path/to/recording.wav -l
```

Outputs land in `{audio_name}_outputs/` next to the audio file. The CSV is the quickest way to inspect results; the `.vocalpy` files let you reload the full recording object in Python for further analysis.

See the [full documentation](https://vocalpy.gumadeiras.com) for CLI reference, output format, and operational checks.
