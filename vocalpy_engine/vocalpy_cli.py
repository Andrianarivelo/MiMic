# -*- coding: utf-8 -*-
"""Backward-compatible wrapper for the packaged VocalPy CLI."""

from vocalpy.cli import build_parser, main


if __name__ == "__main__":
    raise SystemExit(main())
