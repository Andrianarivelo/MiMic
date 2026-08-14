# [VocalPy](https://github.com/gumadeiras/vocalpy)

[VocalPy](https://github.com/gumadeiras/vocalpy) detects, classifies, and segments animal vocalizations from audio recordings. It ships a CLI backed by species-specific detection pipelines, bundled pretrained classifiers, and a autoencoder-based segmentation stage ([SqueakOut](https://github.com/gumadeiras/squeakout)).

The pipeline runs in three stages: **detection** finds candidate vocalizations in the spectrogram using contrast- and morphology-based methods tuned per species; **classification** filters out noise and labels each remaining call by type using pretrained MobileNetV2 models; **segmentation** optionally runs [SqueakOut](https://github.com/gumadeiras/squeakout) to produce a pixel-level binary mask for each detected call. You can run detection alone, detection + classification, or all three.

**Supported species:** mouse, rat, guinea pig

[VocalPy](https://github.com/gumadeiras/vocalpy) is inspired by [VocalMat](https://github.com/ahof1704/VocalMat).

## Start here

- [Getting started](getting-started.md) — installation, CLI reference, output format
- [Example fixtures](example-fixtures.md) — maintained audio fixtures and demo notebooks
- [Operational checks](operations.md) — model validation, baseline checks, docs deploy
- <a href="_source/vocalpy.html">API reference</a>

```{toctree}
:maxdepth: 2
:caption: Documentation

getting-started
example-fixtures
operations
_source/vocalpy
```
