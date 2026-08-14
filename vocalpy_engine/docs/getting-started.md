# Getting started

## Installation

Requires Python 3.12 and [Git LFS](https://git-lfs.github.com/). Git LFS is needed because bundled model checkpoints are stored as LFS objects — if you clone without it the checkpoint files will be stubs and the models will fail to load.

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

## Running the CLI

```sh
vocalpy -p /path/to/recording.wav
```

This runs the full pipeline on a single file: the audio is split into overlapping chunks and processed in parallel, detected vocalizations are filtered for noise and then labeled by type, and all results are written to an output folder next to the audio file. By default the mouse pipeline is used. Pass `-a rat` or `-a guineapig` to switch species.

To process all `.wav` files in a directory at once:

```sh
vocalpy -p /path/to/recordings/
```

Each file gets its own `{name}_outputs/` directory.

## CLI options

| Flag | Description | Default |
|------|-------------|---------|
| `-a / --animal` | Species pipeline: `mouse`, `rat`, `guineapig` | `mouse` |
| `-p / --path_to_audio` | Path to a `.wav` file or a directory of `.wav` files | required |
| `-b / --bin_size` | Audio chunk size in seconds for parallel processing | `60` |
| `-lf / --lower_frequency_cutoff` | Low frequency cutoff in Hz — signals below this are ignored | Species default |
| `-hf / --higher_frequency_cutoff` | High frequency cutoff in Hz — signals above this are ignored | Species default |
| `-t / --threads` | Number of parallel workers (`-1` = half of available cores) | `-1` |
| `-v / --verbose` | Print detailed progress to the terminal | off |
| `-l / --validation` | Save spectrogram-overlay images for manual review of detections | off |
| `--segmenter` | Run autoencoder-based segmentation ([SqueakOut](https://github.com/gumadeiras/squeakout)) after detection and classification | off |
| `--segmentation_model_path` | Path to a custom SqueakOut checkpoint file | Bundled |
| `--segmentation_threshold` | Probability threshold for converting segmentation output to a binary mask | `0.51` |

**Tuning tips:**
- If you're getting too many false positives, try narrowing the frequency range with `-lf` / `-hf` to match what you expect in your recordings.
- For long recordings, increasing `-b` reduces overhead; decreasing it can help on machines with many cores.
- Use `-l` when you're setting up a new recording type or debugging detections — the overlay images show exactly what the detector found on the spectrogram.

## Species defaults

Each species pipeline has tuned spectrogram and detection parameters. The frequency ranges reflect the typical call frequencies for each species. You can override the cutoffs with `-lf` / `-hf` if your recordings differ.

| Species | Frequency range | Window type | Window size | NFFT |
|---------|----------------|-------------|-------------|------|
| mouse | 45,000–125,000 Hz | Hamming | 256 | 1024 |
| rat | 18,000–125,000 Hz | Hamming | 256 | 1024 |
| guineapig | 250–20,000 Hz | Barthann | 512 | 1024 |

**What runs per species:**
- **Mouse and rat:** detection → noise classification (removes non-vocalizations) → type classification (labels each call) → optional segmentation
- **Guinea pig:** detection only — no classifier is available, so all candidates are kept as-is. Neural segmentation via `--segmenter` is still available.

## Output

For a file named `recording.wav`, outputs are written to `recording_outputs/` in the same directory:

```
recording_outputs/
├── recording.wav.csv                       # vocalization metadata table — start here
├── recording_without_spectrograms.vocalpy  # full Recording object, reloadable in Python
├── list_of_vocals.vocalpy                  # ListOfVocals object, reloadable in Python
├── params.yml                              # exact parameters used for this run
├── spectrogram/                            # per-vocal spectrogram image (one PNG per call)
├── mask/                                   # per-vocal binary detection mask (one PNG per call)
├── spectrogram_validation/                 # spectrogram + mask overlay images (-l flag only)
└── cnn_mask/                               # autoencoder-based segmentation masks (--segmenter only)
```

The **CSV** is the fastest way to inspect results — open it in any spreadsheet tool or load it with pandas. The **`.vocalpy` files** let you reload the full pipeline output in Python for further analysis, filtering, or visualization without rerunning detection. The **spectrogram images** are useful for quickly browsing individual calls. The **validation overlays** (`-l`) show the detection mask drawn on top of the spectrogram, which is helpful for understanding what the detector found and catching misdetections.

### CSV columns

Each row is one detected vocalization. For mouse and rat, `top1` and `top2` contain the classifier's type predictions.

| Column | Description |
|--------|-------------|
| `start` / `end` | Absolute time of the call in seconds from the start of the recording |
| `duration` | Call duration in seconds |
| `interval` | Silence between this call and the previous one, in seconds |
| `min_freq` / `max_freq` / `avg_freq` | Lowest, highest, and mean frequency of the call in Hz |
| `bandwidth` | Frequency span of the call (`max_freq - min_freq`) in Hz |
| `min_intensity` / `max_intensity` / `avg_intensity` | Spectrogram intensity range and mean within the detected region |
| `area` | Size of the detected region in spectrogram pixels — larger values mean longer or wider calls |
| `centroid` | Center of mass of the detected region as (time, frequency) coordinates in the spectrogram |
| `orientation` | Angle of the principal axis of the detected region — useful for characterizing call shape |
| `top1` / `top2` | Top-1 and top-2 class label predictions from the type classifier (mouse and rat only) |

## Autoencoder-based segmentation

```sh
vocalpy -p /path/to/recording.wav --segmenter
```

After detection and classification, [SqueakOut](https://github.com/gumadeiras/squeakout) runs on each detected vocal crop and produces a pixel-level binary mask that outlines the vocalization within the spectrogram. This gives finer spatial information than the bounding-box style detection mask. Masks are saved as PNG images under `cnn_mask/`, one per detected call.

- **Input:** each detected call's spectrogram crop is resized to grayscale `1×512×512` before being fed to the model
- **Output:** a binary mask at the same resolution as the crop, where white pixels mark the vocalization
- **Threshold:** the raw model output is a probability map; pixels above the threshold become the mask. The default is `0.51` — lower it to include more of the call boundary, raise it to be more conservative. Override with `--segmentation_threshold`
- **Custom model:** the bundled [SqueakOut](https://github.com/gumadeiras/squeakout) checkpoint is used by default. To use a different checkpoint, pass its path with `--segmentation_model_path`

## Serialized outputs

The `.vocalpy` files are Python objects serialized to disk. They let you reload a prior run in Python without re-running the pipeline:

```python
from vocalpy.utils.io import load_vocalpy_file

recording = load_vocalpy_file("recording_outputs/recording_without_spectrograms.vocalpy")
list_of_vocals = load_vocalpy_file("recording_outputs/list_of_vocals.vocalpy")
```

Files use a versioned envelope format with object-type metadata. Legacy raw-pickle `.vocalpy` files written by older versions load automatically for backward compatibility.

## Packaging notes

- Project metadata: `pyproject.toml`
- Tested dependency pins: `constraints/base.txt` and `constraints/dev.txt`
- Bundled model checkpoints and sidecar metadata: `vocalpy/nn/pretrained/`
- Default pipeline parameters per species: `vocalpy/configs/pipelines_parameters.yml`
