#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rewrite maintained example .vocalpy artifacts through the current envelope."""

from pathlib import Path

from vocalpy.utils.baselines import DEFAULT_MANIFEST, load_fixtures
from vocalpy.utils.io import rewrite_vocalpy_file


def rewrite_fixtures(audio_dir: Path, manifest: Path) -> list[Path]:
    rewritten = []
    for fixture in load_fixtures(manifest):
        recording_path = audio_dir / fixture.outputs_dirname / "recording_without_spectrograms.vocalpy"
        rewrite_vocalpy_file(recording_path, expected_object_type="recording")
        rewritten.append(recording_path)
    return rewritten


def main():
    audio_dir = Path("examples/audios")
    rewritten = rewrite_fixtures(audio_dir=audio_dir, manifest=DEFAULT_MANIFEST)
    for path in rewritten:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
