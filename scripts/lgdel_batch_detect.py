"""Run the upstream MiMic/VocalPy detector on the LgDel close-loop cohort.

This is a thin, resumable adapter around :mod:`vpgui.engine`. It does not change
the detector. It reads the local Pykaboo trial plan, finds each annotated animal
WAV file, skips files that already have an upstream ``*_stats.csv`` output, and
writes a small batch summary after every animal.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time
from dataclasses import asdict, dataclass

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vpgui.engine import DetectParams, run_detection, summarize


@dataclass
class DetectionJob:
    """One animal recording to process with the upstream detector."""

    animal_id: str
    genotype: str
    wav_path: pathlib.Path
    stats_path: pathlib.Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for a resumable cohort detection run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=ROOT.parent,
        help="Folder containing numeric animal recording folders.",
    )
    parser.add_argument(
        "--trial-plan",
        type=pathlib.Path,
        default=ROOT.parent / "pykaboo_trial_plan.csv",
        help="Pykaboo CSV trial plan listing animals to include.",
    )
    parser.add_argument(
        "--summary",
        type=pathlib.Path,
        default=ROOT / "scripts" / "lgdel_detection_summary.csv",
        help="Batch progress CSV written after every animal.",
    )
    parser.add_argument("--animal", default="mouse", choices=("mouse", "rat", "guineapig"))
    parser.add_argument("--bin-size", type=int, default=60)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Recompute files even if stats CSV exists.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of jobs for smoke testing.")
    return parser.parse_args(argv)


def normalize_genotype(value: object) -> str:
    """Normalize local genotype labels into WT or HET."""

    text = "" if value is None else str(value).strip().upper()
    if text.startswith("WT"):
        return "WT"
    if text.startswith("HET") or text in {"HE", "HT"}:
        return "HET"
    return "UNKNOWN"


def load_jobs(data_root: pathlib.Path, trial_plan: pathlib.Path) -> list[DetectionJob]:
    """Build detector jobs from the annotated trial-plan CSV."""

    plan = pd.read_csv(trial_plan)
    jobs: list[DetectionJob] = []
    for _, row in plan.iterrows():
        animal_id = str(row["Animal ID"]).replace(".0", "").strip()
        if not animal_id.isdigit():
            continue
        stem = f"{animal_id}_1_baseline"
        wav_path = data_root / animal_id / "baseline" / "1" / f"{stem}.wav"
        stats_path = data_root / animal_id / "baseline" / "1" / f"{stem}_outputs" / f"{stem}_stats.csv"
        if not wav_path.exists():
            print(f"[skip] {animal_id}: missing WAV {wav_path}", flush=True)
            continue
        jobs.append(
            DetectionJob(
                animal_id=animal_id,
                genotype=normalize_genotype(row.get("genotype")),
                wav_path=wav_path,
                stats_path=stats_path,
            )
        )
    return jobs


def write_summary(summary_path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    """Persist batch progress so an interrupted run can be inspected."""

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_job(job: DetectionJob, params: DetectParams, force: bool) -> dict[str, object]:
    """Run one detection job or report an existing CSV."""

    started = time.perf_counter()
    row: dict[str, object] = {
        **asdict(job),
        "wav_path": str(job.wav_path),
        "stats_path": str(job.stats_path),
        "status": "pending",
        "n_calls": 0,
        "dominant_class": "",
        "duration_s": "",
        "elapsed_s": 0.0,
        "error": "",
    }
    if job.stats_path.exists() and not force:
        try:
            existing = pd.read_csv(job.stats_path)
            row.update(
                {
                    "status": "skipped_existing",
                    "n_calls": int(len(existing)),
                    "duration_s": "",
                    "elapsed_s": 0.0,
                }
            )
            return row
        except Exception as exc:
            print(f"[warn] could not inspect existing CSV for {job.animal_id}: {exc}", flush=True)

    try:
        print(f"[run] {job.animal_id} {job.genotype}: {job.wav_path}", flush=True)
        result = run_detection(
            job.wav_path,
            params,
            phase_cb=lambda msg: print(f"  [{job.animal_id}] {msg}", flush=True),
            log_cb=None,
        )
        stats = summarize(result.detections, result.duration)
        row.update(
            {
                "status": "ok",
                "n_calls": int(stats["n"]),
                "dominant_class": stats["dominant_class"] or "",
                "duration_s": float(result.duration),
                "elapsed_s": round(time.perf_counter() - started, 3),
                "stats_path": str(result.csv_path or job.stats_path),
            }
        )
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "elapsed_s": round(time.perf_counter() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"[fail] {job.animal_id}: {row['error']}", flush=True)
    return row


def main(argv: list[str] | None = None) -> int:
    """Run all missing detections and return success if no job failed."""

    args = parse_args(argv)
    jobs = load_jobs(args.data_root.resolve(), args.trial_plan.resolve())
    if args.limit is not None:
        jobs = jobs[: args.limit]
    params = DetectParams(
        animal=args.animal,
        bin_size=args.bin_size,
        threads=args.threads,
        segmenter=False,
        validation=False,
    )
    print(f"[batch] jobs={len(jobs)} threads={args.threads} bin_size={args.bin_size}", flush=True)
    rows: list[dict[str, object]] = []
    for index, job in enumerate(jobs, start=1):
        print(f"[batch] {index}/{len(jobs)}", flush=True)
        rows.append(run_job(job, params, force=args.force))
        write_summary(args.summary.resolve(), rows)
    failed = [row for row in rows if row["status"] == "failed"]
    print(f"[batch] done, failed={len(failed)} summary={args.summary.resolve()}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
