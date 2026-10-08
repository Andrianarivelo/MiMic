# Injected-call detector recovery

This experiment asks how often the current USV pipeline retains the same known
whistles when they are added to WT and Het recording backgrounds. It measures
detector performance for a fixed challenge, rather than identifying the caller
or estimating the biological loudness of either genotype.

The numerical driver is `scripts/detector_recovery.py`. Its preparation,
detection and reporting stages write to `lgdel_usv_analysis/detector_recovery`.
The independent renderer, `scripts/detector_recovery_figures.py`, reads completed
CSV outputs without importing the detector or changing its numerical cache.

```powershell
python scripts/detector_recovery.py --stage report
python scripts/detector_recovery_figures.py
python scripts/detector_recovery.py --figures-only
```

The first command computes inference from completed recovery trials and renders
the results. The second regenerates plots and the bilingual report from
completed statistics. Expensive audio detection must already be complete.

## Experiment and interpretation

- All 24 recordings contribute genuine 60 s contexts with the original 0.15 s
  forward halo. The planned 360, 540 and 720 s starts were balanced within
  genotype. The zero context at 720 s for Het 29539 was replaced by 360 s.
  Actual allocations are WT 4/4/4 and Het 5/4/3. The replacement still contains
  approximately five initial all-zero seconds. Only active injection slots
  require nonzero background RMS; entire selected contexts are not certified
  as continuously nonzero.
- Four visually checked, cleaned real-whistle templates are shared across every
  recording. Their WT-resident source context does not identify their caller.
  Their approximately 20-65 ms durations do not represent very short ambiguous
  spikes or every possible vocal morphology.
- Received amplitude is band-limited active RMS dBFS. SNR uses the same
  45-125 kHz band and active time support in the local background. Neither is a
  calibrated sound-pressure measurement.
- The main challenge contains all seven levels together. Global normalization
  and segmentation can couple their recoveries. An isolated transition-level
  check quantifies that effect at one level, without validating the entire
  pooled curve as isolated-call recovery.
- Identifier coverage and final retained recovery are separate. One-to-one
  temporal-IoU matching locates injected calls; already retained sham detections
  are excluded from both denominators.
- Recordings are the statistical units. Two normalized curve-AUC comparisons
  share a primary Holm family. Fourteen level comparisons share a secondary
  Holm family. Bootstrap intervals are pointwise and conditional on the fixed
  template library. A nonsignificant result does not establish equivalence.
  Empirical bootstrap bands at [0,0] or [1,1] do not establish population
  impossibility or certainty.

## Delivered artifacts

The output directory contains the complete injection, template and background
manifests; individual recovery trials; recording means; primary AUC and secondary
level statistics; the isolated sensitivity; provenance; and validation checks.
`detection_stage_summary.csv` gives equal-recording identifier and retained
probabilities by level, genotype and variant, together with injected, eligible
and sham-excluded counts. The report also compares the descriptive recovery
peak with the highest tested level and gives paired isolated-minus-pooled
changes. These summaries introduce no additional significance tests.

`detector_recovery_curves` is the compact main figure. Supplemental figures cover
identifier recovery, isolated-versus-pooled recovery, measured background levels,
and original-versus-cleaned template spectra. Every figure is saved as PNG, PDF,
and SVG. `RAPPORT_RECUPERATION_DETECTEUR.md` gives the French and English results,
assumptions, limits, and links to all statistical tables.

Original recordings, metadata, and preceding behavioral analyses are unchanged.

## Literal zero audio and missing exposure

The independent 300-900 s audio audit reports 254 complete all-zero seconds for
WT 29999, 129 for WT 30000, 244 for Het 29539, and seven for Het 31101. The latter
also lacks approximately 2.43 s of audio. WT 31337 and 31315 each have one
complete all-zero second. Individual zero-valued samples can be normal PCM
quantization and are not automatically considered dropouts.

The background supplement includes a coarse availability display: read audio
duration minus complete all-zero seconds, relative to the 600 s window. It does
not certify every remaining sample as usable, nor detect every partial silent
interval. Full-context RMS still includes natural sounds and zero-valued parts.

Earlier behavioral vocalization analyses used video-based exposure rather than
the intersection of valid audio and video. This calibration does not correct
their rates, probabilities, correlations, or ANOVA. A separate reanalysis must
establish audio exposure before treating absent calls as observed silence.

Use `scripts/detector_recovery.cmd` with its configured `mamir` environment for
detection. On this workspace, its interpreter is
`C:\Users\andry\miniconda3\envs\mamir\python.exe`; the launcher resolves the
same location through `%USERPROFILE%` and falls back to `python` if absent.

```powershell
.\scripts\detector_recovery.cmd --stage detect
```

The renderer and reporting code can run in the base scientific environment once
all detection outputs are complete; they do not import or require PyTorch.

## Separate intact-context sensitivity

The original 29539 replacement at 360 s includes approximately five seconds of
literal zero audio. The unchanged detector computes logarithmic spectrograms;
zeros can produce negative infinity and invalid normalization, even when
injections themselves occupy nonzero slots. A separate experiment uses its
540-600.15 s context without silently changing the primary experiment.

`scripts/detector_recovery_context_sensitivity.py` prepares this recording only.
`scripts/detector_recovery_context_sensitivity.cmd` provides the corresponding
double-click launcher; its default also prepares only.
Its temporary driver configuration and genotype map are restored on exit. It
preserves the original four-template/seven-level randomized schedule. Default
execution prepares files only and does not start another detector worker.

```powershell
python scripts/detector_recovery_context_sensitivity.py --stage prepare
& "$env:USERPROFILE\miniconda3\envs\mamir\python.exe" scripts/detector_recovery_context_sensitivity.py --stage detect
python scripts/detector_recovery_context_sensitivity.py --stage combine
```

Start detection after a primary worker becomes free. Run `combine` only after
all 24 primary recordings and the five sensitivity chunks finish. It writes a
new `intact_context_sensitivity/combined_24_recordings` directory, replacing
only the derived rows for 29539. Original files and primary results are retained.
The two re-estimated AUC contrasts form a separate exploratory Holm family.
This context check was chosen after the zero-audio problem was discovered and
is not independent confirmation.

The observed retained curves are nonmonotonic despite high identifier coverage
at stronger levels. Their loss appears at final noise-classifier selection;
upstream changes to masks and image contrast may influence that decision. The
isolated transition check improves recovery but does not provide a complete
isolated-call curve. Neither nonsignificant genotype tests nor these controls
exclude quieter Het emission, microphone-distance differences, or untested call
morphology. No equivalence margin or equivalence test was specified.

Detection resumes per-file caches by default. `--recompute --stage detect`
repeats detection from validated generated injection WAVs and replaces only
named derived prediction caches. Use `--output-dir <new-directory>` when
changing numerical experiment settings, preserving previous results.
`provenance.json` fingerprints injection/detection inputs separately from
`analysis_provenance.json`, which records the inference code, renderer,
bootstrap/permutation counts and runtime versions.
