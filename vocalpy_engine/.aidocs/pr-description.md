Problem

The docs toolchain still depended on `recommonmark`, which was already emitting
parser warnings, and the generated docs tree still exposed private test modules
as if they were part of the supported public surface.

Mental model

The docs site should use a maintained Markdown parser, keep public API pages in
the docs tree, and avoid publishing internal test modules as product
documentation. The generated docs tree should match the actual maintained
surface: package, modules, pipelines, and utilities.

Non-goals

- No runtime behavior changes
- No notebook/example changes in this slice
- No theme redesign

Tradeoffs

The public docs tree is smaller now, which improves signal, but it also means
internal test-package autodoc pages are no longer available from the built
site.

Architecture

- `recommonmark` was replaced by `myst-parser` in the docs dependency set
- `docs/conf.py` now uses explicit Sphinx extensions:
  - `myst_parser`
  - `sphinx.ext.autodoc`
  - `sphinx.ext.napoleon`
  - `sphinx_rtd_theme`
- the old orphan/test rst pages were removed from `docs/_source/`
- `docs/index.md` now owns the docs landing page and links the API reference
  tree through a hidden toctree

Observability

- `python -m sphinx -b html docs docs/_build/html`
- verify the build no longer reports `recommonmark` warnings
- inspect `docs/_build/html/index.html` and the generated module index

Tests

- `python -m sphinx -b html docs docs/_build/html`
- `python -m pytest -q`
