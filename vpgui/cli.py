"""Headless CLI: detect USVs in files/folders and write a per-file CSV.

Drives the same :mod:`vpgui.engine` path as the GUI, so the results match. Each
input file yields ``{stem}_outputs/{stem}_stats.csv`` (plus SqueakOut masks when
``--segmenter`` is set) next to the audio, exactly like the upstream tool.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from . import engine, spectro
from .engine import DetectParams


def _expand(paths, recursive):
    files = []
    for p in paths:
        p = pathlib.Path(p)
        if p.is_dir():
            files.extend(spectro.find_audio_files(p, recursive=recursive))
        elif p.is_file():
            files.append(p)
    seen, out = set(), []
    for f in files:
        if str(f) not in seen:
            seen.add(str(f))
            out.append(f)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vocalpy-cli",
        description="Detect, classify and (optionally) segment ultrasonic "
                    "vocalizations, writing a per-file *_stats.csv.")
    p.add_argument("paths", nargs="+", help="audio file(s) and/or folder(s)")
    p.add_argument("-a", "--animal", default="mouse", choices=list(engine.SPECIES))
    p.add_argument("-b", "--bin-size", type=int, default=60, help="chunk size (s)")
    p.add_argument("-lf", "--lower-freq", default=None, type=int, help="lower cutoff (Hz)")
    p.add_argument("-hf", "--higher-freq", default=None, type=int, help="higher cutoff (Hz)")
    p.add_argument("-t", "--threads", type=int, default=-1, help="-1 = cores/2")
    p.add_argument("--segmenter", action="store_true", help="run SqueakOut segmentation")
    p.add_argument("-l", "--validation", action="store_true", help="save overlay images")
    p.add_argument("--no-recursive", action="store_true", help="do not descend into sub-folders")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    files = _expand(args.paths, recursive=not args.no_recursive)
    if not files:
        print("no audio files found in the given paths", file=sys.stderr)
        return 2

    params = DetectParams(
        animal=args.animal,
        bin_size=args.bin_size,
        lower_frequency_cutoff="default" if args.lower_freq is None else args.lower_freq,
        higher_frequency_cutoff="default" if args.higher_freq is None else args.higher_freq,
        threads=args.threads,
        segmenter=args.segmenter,
        validation=args.validation,
    )
    print(f"detecting {args.animal} USVs in {len(files)} file(s)"
          + (" + SqueakOut" if args.segmenter else ""))
    n_ok = n_fail = 0
    for i, f in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {f}")
        try:
            result = engine.run_detection(f, params, phase_cb=lambda m: print(f"    {m}"))
            s = engine.summarize(result.detections, result.duration)
            print(f"    -> {s['n']} calls · top {s['dominant_class']} · CSV: {result.csv_path}")
            n_ok += 1
        except Exception as exc:
            print(f"    ! failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            n_fail += 1
    print(f"done: {n_ok} ok, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
