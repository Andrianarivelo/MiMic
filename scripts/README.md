# Example application: session-6 USV analysis

This directory is a complete example of taking VocalPy detection output into a
small research analysis. It compares ultrasonic vocalizations from two WT and
four Het resident mice across a 15-minute social-interaction session:

- `alone`: 0–300 seconds
- `female`: 300–600 seconds, after a female is introduced
- `male`: 600–900 seconds, after a second male is introduced

The example builds per-call spectrogram cards and embeddings, then analyzes
call rate, syllable repertoire, acoustic properties, embedding geometry, and
exploratory voice attribution. It is intentionally study-specific rather than
a generic command-line interface.

## Pipeline

Run the scripts in numeric order:

```bash
python scripts/00_build_dataset.py
python scripts/01_call_rate.py
python scripts/02_repertoire.py
python scripts/03_acoustics.py
python scripts/04_embedding.py
python scripts/05_voice_attribution.py
```

| Script | Purpose | Main output |
| --- | --- | --- |
| `00_build_dataset.py` | Load VocalPy CSVs, extract fixed-size spectrogram cards, and compute DINOv2/ResNet18 embeddings | `calls_dataset.npz`, `calls_meta.csv`, QC montage |
| `01_call_rate.py` | Compare calls per minute across genotype and phase | Call-rate, timeline, and total-output figures |
| `02_repertoire.py` | Compare the 11-class syllable repertoire | Stacked repertoire, heatmap, and count figures |
| `03_acoustics.py` | Compare duration, frequency, bandwidth, and intensity features | Violin, phase, and duration-frequency figures |
| `04_embedding.py` | Visualize embedding geometry and decode mouse/genotype identity | UMAP, confusion-matrix, and cross-phase figures |
| `05_voice_attribution.py` | Explore newcomer-call novelty and phase-associated clusters | Attribution UMAP, novelty, and cluster figures |
| `common.py` | Define the cohort, phase boundaries, paths, plotting style, and shared helpers | Imported by every analysis module |

Generated figures are written to `/home/andry/UVS/figures/`. The cached dataset
and metadata are written beside these scripts. Generated caches are ignored by
Git and should be rebuilt locally.

## Prerequisites

Start with the repository environment described in
[`environment.yaml`](../environment.yaml). The analysis scripts additionally
use:

- `scikit-learn`
- `umap-learn`
- Pillow
- the NEMBA frame-embedding code used by `00_build_dataset.py`

The dataset builder currently imports NEMBA from
`/home/andry/tracking_project/NEMBA` and requests the `dinov2_s` and `resnet18`
frame embedders. If an embedder is unavailable, the builder reports the failure
and continues, but scripts that require `emb_dinov2` cannot run without that
embedding.

## Expected input layout

This example expects the study data under `/home/andry/UVS/dataset` using this
shape:

```text
dataset/
└── <mouse-id>/
    └── 6/
        ├── <recording>.wav
        └── <recording>_outputs/
            └── <recording>_stats.csv
```

The six mouse IDs, genotypes, and recording names are declared in `common.py`.
The `*_stats.csv` files are the per-file detection exports produced by the
VocalPy GUI or CLI. To adapt the example to another cohort, edit `DATASET`,
`FIGURES`, `FILES`, and the phase definitions in `common.py`.

## Reproducing the example

1. Detect each source WAV with the VocalPy workbench and retain its
   `<recording>_stats.csv` output beside the recording.
2. Verify that `FILES` in `common.py` matches the local mouse IDs, genotypes,
   recording names, and session number.
3. Run `00_build_dataset.py` to create the shared per-call cache.
4. Run modules 01–05 to generate the statistical summaries and figures.
5. Treat the printed caveats as part of the result: the example has only two WT
   and four Het animals, highly imbalanced call counts, and no ground-truth
   speaker labels.

## Interpretation limits

This is an example analysis, not a validated biological conclusion. Statistical
tests operate at animal level where possible, but the cohort is underpowered.
Some low-confidence Het detections may be broadband noise, and the voice
attribution module is hypothesis-generating because a single microphone cannot
provide true caller identity.
