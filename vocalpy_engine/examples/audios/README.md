# Example Audio Fixtures

`mouse_1.wav` and `mouse_2.wav` plus their matching `*_outputs/` directories are
 the maintained example fixtures. They back the exact baseline smoke checks in CI.

Their `.vocalpy` artifacts are expected to use the current versioned VocalPy
serialization envelope. Legacy raw-pickle `.vocalpy` files still load, but new
fixtures should always be written by the package helpers so the object type is
recorded alongside the payload. The repo maintenance helper for refreshing the
checked-in fixture artifacts is `python scripts/rewrite_example_vocalpy_fixtures.py`.

`example_outputs/` is legacy image-only demo data kept for the classifier notebooks.
It is not part of the CI baseline manifest and should not be treated as canonical
pipeline output for the modernized package.
