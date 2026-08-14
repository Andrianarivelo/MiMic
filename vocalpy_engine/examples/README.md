# Example Notebooks

This directory mixes a small maintained notebook surface with older research
scratchpads. They should not all be treated as equally current.

## Maintained notebooks

- `class_classifier_demo.ipynb`: bundled class-model demo against the shipped
  example spectrogram images
- `noise_classifier_demo.ipynb`: bundled noise-model demo against the shipped
  example spectrogram images
- `visualization_plots_demo.ipynb`: visualization API demo using the maintained
  `mouse_1` and `mouse_2` recording fixtures

## Developer notebook

- `debug.ipynb`: ad hoc inspection notebook for local debugging of shipped
  fixtures and spectrogram images

## Archived notebooks

- `candidate_vocalization_identifier_demo.ipynb`
- `unsupervised_embedding.ipynb`
- `unsupervised_embedding-Copy1.ipynb`

Archived notebooks are kept for historical context only. They may depend on
older APIs, placeholder datasets, or exploratory code paths that are not part
of the maintained package surface.

## Notebook hygiene

- committed notebooks should load as valid JSON
- committed notebooks should not keep stale execution outputs or machine-
  specific tracebacks
- maintained fixture paths should stay relative to `examples/`
