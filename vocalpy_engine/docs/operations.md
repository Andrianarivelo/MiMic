# Operational checks

## Bundled models

VocalPy ships three pretrained models that run during the analysis pipeline. The `noise` and `class` models are MobileNetV2 classifiers that operate on per-vocal spectrogram crops. The `segment` model is [SqueakOut](https://github.com/gumadeiras/squeakout), which produces pixel-level binary masks. All three are packaged as checkpoints under `vocalpy/nn/pretrained/` with sidecar JSON metadata that records their architecture, expected input shape, output classes, SHA-256 checksum, and provenance.

| Type | Architecture | Role in pipeline |
|------|-------------|-----------------|
| `noise` | MobileNetV2 | Runs first after detection — classifies each candidate as vocal or noise and removes noise candidates |
| `class` | MobileNetV2 | Runs second — assigns a vocalization-type label to each remaining call |
| `segment` | [SqueakOut](https://github.com/gumadeiras/squeakout) (MobileNetV2 backbone) | Runs last when `--segmenter` is passed — generates a binary mask for each call |

## Validate bundled models

Use `vocalpy-models` to inspect and validate the bundled checkpoints. This is useful after cloning or pulling to confirm the LFS files were fetched correctly, or when updating a checkpoint to verify the new file matches expectations.

List metadata for all bundled checkpoints:

```sh
vocalpy-models list
```

Validate checksums to confirm the on-disk files match their expected SHA-256 digests:

```sh
vocalpy-models validate
```

Also load the checkpoint and run a deterministic dummy forward pass on CPU — catches corrupt or mismatched weights:

```sh
vocalpy-models validate --smoke-test
```

Target a specific model type instead of all three:

```sh
vocalpy-models list --network-type noise
vocalpy-models validate --network-type segment --smoke-test
```

Available types: `noise`, `class`, `segment`, `all` (default).

Emit machine-readable JSON — useful for scripting or CI assertions:

```sh
vocalpy-models list --json
vocalpy-models validate --smoke-test --json
```

## Validate example fixtures

The maintained mouse fixtures (`mouse_1.wav`, `mouse_2.wav`) have stored expected outputs. Running the comparison script processes both files through the current pipeline and checks whether the results match the stored baseline — number of vocalizations, timing, classification labels, etc.

```sh
python scripts/compare_example_baselines.py --species mouse
```

Use this after changing detection or classification logic to confirm you haven't introduced an unintended regression. The fixture manifest lives in `examples/audios/baselines.yml`.

## Rewrite fixtures

When a pipeline change intentionally shifts expected outputs (for example, after tuning detection thresholds or updating a model), rewrite the serialized fixture files so future comparisons use the new expected values:

```sh
python scripts/rewrite_example_vocalpy_fixtures.py
```

Run the comparison script afterward to confirm the rewritten fixtures now pass.

## Reference

- <a href="genindex.html">General index</a>
- <a href="py-modindex.html">Python module index</a>
