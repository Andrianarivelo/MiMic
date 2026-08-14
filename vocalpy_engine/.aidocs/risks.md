- `docs/conf.py` still imports `vocalpy.__version__`.
  Why it matters: if `vocalpy/__init__.py` ever grows import-heavy side effects,
  docs build could become more fragile.
  Smallest documentation fix: keep `vocalpy.__init__` lightweight by convention.
  Follow-up code fix: if that changes, read version from `pyproject.toml`
  without importing the package.

- The docs landing page links to GitHub markdown for some higher-level repo
  guidance instead of duplicating that content inside Sphinx.
  Why it matters: offline docs readers will not see that material in the built
  site itself.
  Smallest documentation fix: add dedicated Sphinx pages later if offline docs
  become important.
  Follow-up code fix: none.

- Internal test-package autodoc pages were removed from the docs tree.
  Why it matters: contributors who relied on the rendered test-package pages now
  need to read the source directly.
  Smallest documentation fix: add a contributor docs page later if test layout
  guidance becomes necessary.
  Follow-up code fix: none.
