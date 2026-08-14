"""Shared foundation for the session-6 USV analysis.

Experimental design (session 6, 15 min = 900 s), 3 phases of 5 min each:
  * alone   [  0-300 s): resident male alone
  * female  [300-600 s): a female is introduced
  * male    [600-900 s): a second (intruder) male is added

Genotypes: WT = {31096, 31097} (blue / C0), Het = {31098,31099,31101,31102} (orange / C1).

This module owns the constants, palette, phase assignment, data loading, and the
per-call spectrogram "card" extraction used everywhere downstream.
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import spectrogram, get_window

DATASET = pathlib.Path("/home/andry/UVS/dataset")
SCRIPTS = pathlib.Path("/home/andry/UVS/scripts")
FIGURES = pathlib.Path("/home/andry/UVS/figures")
FIGURES.mkdir(exist_ok=True)

# --- colours: WT = blue (C0), Het = orange (C1) -----------------------------
WT_COLOR = "#1f77b4"     # C0
HET_COLOR = "#ff7f0e"    # C1
GENO_COLOR = {"WT": WT_COLOR, "Het": HET_COLOR}
PHASE_ORDER = ["alone", "female", "male"]
PHASE_LABEL = {"alone": "alone", "female": "+ female", "male": "+ 2nd male"}
PHASE_BOUNDS = (300.0, 600.0, 900.0)

# --- session-6 files: mouse -> (genotype, wav, stats.csv) -------------------
def _f(mid, geno, sub):
    wav = DATASET / f"{mid}/6/{sub}.wav"
    csv = DATASET / f"{mid}/6/{sub}_outputs/{sub}_stats.csv"
    return dict(mouse=mid, geno=geno, wav=wav, csv=csv)

FILES = [
    _f("31096", "WT", "31096_6_UVS_F_basal"),
    _f("31097", "WT", "31097_6_UVS_F_basal"),
    _f("31098", "Het", "31098_6_UVS_F"),
    _f("31099", "Het", "31099_6_UVS_F"),
    _f("31101", "Het", "31101_6_UVS_F_basal"),
    _f("31102", "Het", "31102_6_UVS_F_basal"),
]

# 11-class syllable palette (matches vpgui.engine.CLASS_COLORS ordering intent)
SYLLABLE_CLASSES = ["chevron", "complex", "down_fm", "flat", "mult_steps",
                    "rev_chevron", "short", "step_down", "step_up", "two_steps", "up_fm"]

# spectrogram params (mouse pipeline) for the call cards
NFFT, WIN, NOVERLAP = 1024, 256, 128
CARD_BAND = (20_000.0, 125_000.0)   # Hz shown in each call card
CARD_WIN_S = 0.12                   # fixed window (s) centred on each call
CARD_SIZE = 128                     # output card is CARD_SIZE x CARD_SIZE


def phase_of(t: float) -> str:
    return "alone" if t < 300 else ("female" if t < 600 else "male")


def load_calls() -> pd.DataFrame:
    """Concatenate all session-6 detection CSVs with mouse/geno/phase columns."""
    frames = []
    for e in FILES:
        d = pd.read_csv(e["csv"])
        d["mouse"] = e["mouse"]
        d["geno"] = e["geno"]
        d["phase"] = d["start(s)"].apply(phase_of)
        d["dur_ms"] = d["duration(ms)"]
        d["avg_khz"] = d["avg_freq"] / 1000.0
        d["min_khz"] = d["min_freq"] / 1000.0
        d["max_khz"] = d["max_freq"] / 1000.0
        d["bw_khz"] = d["bandwidth"] / 1000.0
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["phase"] = pd.Categorical(df["phase"], categories=PHASE_ORDER, ordered=True)
    return df


# --- acoustic feature vector (for the handcrafted-feature embedding) --------
FEATURE_COLS = ["dur_ms", "avg_khz", "min_khz", "max_khz", "bw_khz",
                "avg_intensity", "bg_intensity", "contrast_db", "area(pixels)", "interval(s)"]


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    d = df.copy()
    d["contrast_db"] = d["avg_intensity"] - d["bg_intensity"]
    X = d[FEATURE_COLS].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # log-scale the heavy-tailed magnitude features
    for j, c in enumerate(FEATURE_COLS):
        if c in ("dur_ms", "area(pixels)", "bw_khz"):
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    return X


# --- per-call spectrogram card ---------------------------------------------
def _robust_uint8(img: np.ndarray) -> np.ndarray:
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img, dtype=np.uint8)
    lo = np.percentile(finite, 60)   # background -> dark
    hi = np.percentile(finite, 99.5)
    hi = hi if hi > lo else lo + 1.0
    z = np.clip((img - lo) / (hi - lo), 0, 1)
    return (z * 255).astype(np.uint8)


def call_card(samples: np.ndarray, sr: int, denoise: bool = True) -> np.ndarray:
    """Compute a fixed-size grayscale spectrogram card (CARD_SIZE^2, uint8)."""
    from PIL import Image
    f, t, Pxx = spectrogram(samples, fs=sr, window=get_window("hamming", WIN),
                            noverlap=NOVERLAP, nfft=NFFT, mode="psd")
    sel = (f >= CARD_BAND[0]) & (f <= CARD_BAND[1])
    S = Pxx[sel]
    with np.errstate(divide="ignore"):
        S = 10.0 * np.log10(np.maximum(S, 1e-20))
    if denoise:
        base = np.nanmedian(np.where(np.isfinite(S), S, np.nan), axis=1, keepdims=True)
        S = S - np.nan_to_num(base)
    img = _robust_uint8(S)
    img = np.flipud(img)   # low freq at bottom
    return np.asarray(Image.fromarray(img).resize((CARD_SIZE, CARD_SIZE), Image.BILINEAR))


def extract_cards(df: pd.DataFrame) -> np.ndarray:
    """Extract a call card for every row of ``df`` (must have mouse/start/end)."""
    wav_by_mouse = {e["mouse"]: e["wav"] for e in FILES}
    info_cache = {}
    cards = np.zeros((len(df), CARD_SIZE, CARD_SIZE), dtype=np.uint8)
    for i, (_, r) in enumerate(df.iterrows()):
        wav = wav_by_mouse[r["mouse"]]
        if wav not in info_cache:
            info_cache[wav] = sf.info(str(wav))
        info = info_cache[wav]
        sr = info.samplerate
        centre = 0.5 * (r["start(s)"] + r["end(s)"])
        half = CARD_WIN_S / 2.0
        start = int(max(0, round((centre - half) * sr)))
        n = int(round(CARD_WIN_S * sr))
        n = min(n, info.frames - start)
        x, _ = sf.read(str(wav), start=start, frames=n, always_2d=True)
        cards[i] = call_card(x[:, 0], sr)
    return cards


def apply_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#333333", "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 11, "axes.titlesize": 12,
        "axes.titleweight": "bold", "legend.frameon": False,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "font.family": ["Ubuntu", "DejaVu Sans"],
    })


def mannwhitney(a, b):
    """Return (U, p, rank-biserial effect size) for two samples; NaNs dropped."""
    from scipy.stats import mannwhitneyu
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 1 or len(b) < 1:
        return np.nan, np.nan, np.nan
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    rbc = 2 * U / (len(a) * len(b)) - 1   # rank-biserial correlation
    return U, p, rbc


def pstars(p):
    if not np.isfinite(p):
        return "n/a"
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def sig_bracket(ax, x1, x2, y, p, h=None):
    h = h if h is not None else (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.1, c="#333333")
    ax.text((x1 + x2) / 2, y + h, pstars(p), ha="center", va="bottom", fontsize=11)


def save_fig(fig, name: str, dpi: int = 150):
    out = FIGURES / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  saved {out}")
    return out
