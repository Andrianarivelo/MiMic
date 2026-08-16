"""Shared configuration and loaders for the single-mic USV call-attribution pipeline.

Data layout (one session per experimental animal):
  D:/LgDel_close_loop/<animal>/baseline/1/<animal>_1_baseline.wav        384 kHz mono
  .../<animal>_1_baseline_tracking_dlc.csv    mouse1=resident, mouse2=partner (lik<0 => absent)
  .../<animal>_1_baseline_live_detections.csv per-frame behavior flags
Phases: alone 0-300 s (only the resident can call), partner 300-900 s (either animal).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DATA_ROOT = pathlib.Path(r"D:\LgDel_close_loop")
MIMIC_ROOT = DATA_ROOT / "MiMic"
MERGED_CALLS = MIMIC_ROOT / "lgdel_usv_analysis" / "detected_calls_merged.csv"
OUT_DIR = MIMIC_ROOT / "lgdel_usv_analysis" / "attribution"

PARTNER_INTRO_S = 300.0
SESSION_END_S = 900.0
FPS = 30.0
SR = 384_000

# Stim (partner) animal per experimental animal, from pykaboo_trial_plan_completed.xlsx
# "Comments" column. None = not recorded in the plan. 51557 and 51558 are reused
# across two sessions each, which the attribution model exploits.
STIM_MAP: dict[str, str | None] = {
    "31337": "51790", "31336": "51791", "31335": "51792", "31334": "51793",
    "31333": "51789", "31097": "51794", "31100": "51795", "31101": "51796",
    "31102": "51592", "31077": "51557", "31078": "51558", "31314": "51594",
    "31315": "51557", "31316": "51558", "31317": "51559", "31318": "51560",
    "31319": "51561", "31076": None, "31075": None, "29994": None,
    "29999": None, "30000": None, "29538": None, "29539": None,
}

# Canonical genotypes from pykaboo_trial_plan_completed.xlsx `genotype_pyrat`
# (authoritative; the older plan CSV has 31097/31101 flipped).
GENOTYPE_MAP: dict[str, str] = {
    "31337": "WT", "31336": "WT", "31097": "WT", "31102": "WT", "31078": "WT",
    "31315": "WT", "31316": "WT", "31317": "WT", "31318": "WT", "29994": "WT",
    "29999": "WT", "30000": "WT",
    "31335": "HET", "31334": "HET", "31333": "HET", "31100": "HET",
    "31101": "HET", "31077": "HET", "31076": "HET", "31075": "HET",
    "31314": "HET", "31319": "HET", "29538": "HET", "29539": "HET",
}


def apply_genotype(df: pd.DataFrame) -> pd.DataFrame:
    """Overwrite the genotype column with the authoritative XLSX labels."""
    df = df.copy()
    df["genotype"] = df["animal_id"].astype(str).map(GENOTYPE_MAP)
    return df


# Spectrogram settings for feature extraction (384 kHz mono).
NPERSEG = 512          # 750 Hz frequency resolution
NOVERLAP = 384         # 128-sample hop -> 3000 frames/s
F_LO = 40_000.0
F_HI = 125_000.0
RIDGE_SNR_DB = 6.0     # frame kept in the ridge if peak SNR above noise floor
CROP_SIZE = 64         # AE input crops are CROP_SIZE x CROP_SIZE

RNG_SEED = 20260814


def session_dir(animal: str) -> pathlib.Path:
    return DATA_ROOT / str(animal) / "baseline" / "1"


def wav_path(animal: str) -> pathlib.Path:
    return session_dir(animal) / f"{animal}_1_baseline.wav"


def tracking_path(animal: str) -> pathlib.Path:
    return session_dir(animal) / f"{animal}_1_baseline_tracking_dlc.csv"


def live_detections_path(animal: str) -> pathlib.Path:
    return session_dir(animal) / f"{animal}_1_baseline_live_detections.csv"


def load_calls() -> pd.DataFrame:
    """All detections for the 24 annotated animals, rebuilt from the raw
    per-session stats CSVs.

    NOTE: `detected_calls_merged.csv` in lgdel_usv_analysis is truncated at
    600 s (it silently drops every call from the last 5 minutes - 564 calls,
    including 225 of 31337's 300). We therefore go back to the raw
    ``*_stats.csv`` files, which cover the full 0-900 s session.
    """
    frames = []
    for animal in GENOTYPE_MAP:
        p = (session_dir(animal) / f"{animal}_1_baseline_outputs"
             / f"{animal}_1_baseline_stats.csv")
        d = pd.read_csv(p)
        d = d.rename(columns={
            "start(s)": "start_s", "end(s)": "end_s",
            "duration(ms)": "duration_ms", "area(pixels)": "area_px",
        })
        d = d.drop(columns=[c for c in d.columns if c.startswith("Unnamed")])
        d["animal_id"] = str(animal)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["phase"] = np.where(df["start_s"] < PARTNER_INTRO_S, "alone", "partner")
    df = df.sort_values(["animal_id", "start_s"]).reset_index(drop=True)
    df["call_id"] = df["animal_id"] + "_" + df.groupby("animal_id").cumcount().astype(str)
    return apply_genotype(df)


def load_tracking(animal: str) -> pd.DataFrame:
    """Tracking as columns t, m1_x, m1_y, m1_lik, m2_x, m2_y, m2_lik.

    Low/negative likelihood entries (absent or unreliable animal) become NaN.
    """
    # DLC-style CSVs carry 4 header rows: scorer, individuals, bodyparts, coords
    raw = pd.read_csv(tracking_path(animal), skiprows=4, header=None, low_memory=False)
    out = pd.DataFrame({
        "t": raw.iloc[:, 1].astype(float),
        "m1_x": raw.iloc[:, 2].astype(float),
        "m1_y": raw.iloc[:, 3].astype(float),
        "m1_lik": raw.iloc[:, 4].astype(float),
        "m2_x": raw.iloc[:, 29].astype(float),
        "m2_y": raw.iloc[:, 30].astype(float),
        "m2_lik": raw.iloc[:, 31].astype(float),
    })
    for m in ("m1", "m2"):
        bad = ~(out[f"{m}_lik"] > 0.2)
        out.loc[bad, [f"{m}_x", f"{m}_y"]] = np.nan
    return out


def interp_track(track: pd.DataFrame, times: np.ndarray) -> dict[str, np.ndarray]:
    """Linear interpolation of both animals' positions and speeds at call times."""
    t = track["t"].to_numpy()
    res: dict[str, np.ndarray] = {}
    for m in ("m1", "m2"):
        for ax in ("x", "y"):
            v = track[f"{m}_{ax}"].to_numpy()
            ok = np.isfinite(v)
            if ok.sum() < 2:
                res[f"{m}_{ax}"] = np.full_like(times, np.nan, dtype=float)
                continue
            vi = np.interp(times, t[ok], v[ok])
            # invalidate interpolation across gaps larger than 0.5 s
            gap = np.full_like(times, np.inf)
            idx = np.searchsorted(t[ok], times).clip(1, ok.sum() - 1)
            left = t[ok][idx - 1]
            right = t[ok][idx]
            gap = np.maximum(np.abs(times - left), np.abs(right - times))
            vi[gap > 0.5] = np.nan
            res[f"{m}_{ax}"] = vi
        # speed via +-0.15 s finite difference on the interpolated trace
        dt = 0.15
        x0 = np.interp(times - dt, t, track[f"{m}_x"].to_numpy())
        x1 = np.interp(times + dt, t, track[f"{m}_x"].to_numpy())
        y0 = np.interp(times - dt, t, track[f"{m}_y"].to_numpy())
        y1 = np.interp(times + dt, t, track[f"{m}_y"].to_numpy())
        res[f"{m}_speed"] = np.hypot(x1 - x0, y1 - y0) / (2 * dt)
    res["dist12"] = np.hypot(res["m1_x"] - res["m2_x"], res["m1_y"] - res["m2_y"])
    return res


def ensure_out() -> pathlib.Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR
