#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compatibility wrapper for mouse-only baseline validation."""

import sys

from vocalpy.utils.baselines import main


if __name__ == "__main__":
    raise SystemExit(main(["--species", "mouse", *sys.argv[1:]]))
