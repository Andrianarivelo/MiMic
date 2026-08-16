"""V2 attribution core: independent per-call cues -> fused log-odds -> HMM.

Cues (all trained WITHOUT stim IDs and WITHOUT pretrained checkpoints):
  KIN    caller-kinematics likelihood ratio. A logistic "does this animal's
         movement look like call emission?" model is trained on alone-phase
         ground truth (gate: LOAO AUC 0.65); per call the resident's and
         partner's scores are differenced.
  VOICE  anchored two-source mixture log-likelihood ratio in a chosen
         embedding space (v1 AE space or v2 bout-contrastive space), partner
         component free per session (no stim tying).
  LEVEL  received level with intrinsic loudness regressed out (call-shape
         covariates), residual tested against a distance law. Auto-disabled
         if the leave-one-animal-out gate fails.
  OVLP   two-voice score: fraction of call frames showing two simultaneous
         non-harmonic frequency bands (direct evidence of a second caller;
         used for validation/abstention, not caller choice).

Fusion: logistic regression on cue log-odds (leave-pair-out benchmarked),
followed by the gap-dependent caller HMM from v1 (soft bout aggregation).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A
import attr_core as CORE

KIN_FEATS = ["speed", "accel", "turn"]
# speed sampled around the call: emission sits inside a locomotor envelope
# (caller speeds up into the call), so lags carry extra caller information
KIN_LAGS = [-1.0, -0.5, 0.5, 1.0]
KIN_DIM = len(KIN_FEATS) + len(KIN_LAGS)


def kin_feature_matrix(k: pd.DataFrame, mouse: str, times: np.ndarray,
                       phase: str) -> np.ndarray:
    """Extended kinematic features for one animal at given times, z-scored
    with that animal's session-phase statistics."""
    t = k["t"].to_numpy()
    m = t < A.PARTNER_INTRO_S if phase == "alone" else t >= A.PARTNER_INTRO_S
    cols = []
    for f in KIN_FEATS:
        v = k[f"{mouse}_{f}"].to_numpy()
        mu, sd = np.nanmean(v[m]), np.nanstd(v[m])
        ok = np.isfinite(v)
        vi = (np.interp(times, t[ok], v[ok]) if ok.sum() > 10
              else np.full(len(times), np.nan))
        cols.append((vi - mu) / (sd + 1e-9))
    v = k[f"{mouse}_speed"].to_numpy()
    mu, sd = np.nanmean(v[m]), np.nanstd(v[m])
    ok = np.isfinite(v)
    for lag in KIN_LAGS:
        vi = (np.interp(times + lag, t[ok], v[ok]) if ok.sum() > 10
              else np.full(len(times), np.nan))
        cols.append((vi - mu) / (sd + 1e-9))
    return np.stack(cols, axis=1)


# ------------------------------------------------------------------ loading
def load_kinematics():
    d = np.load(A.OUT_DIR / "v2_kinematics.npz", allow_pickle=True)
    cols = [str(c) for c in d["columns"]]
    kin = {str(a): pd.DataFrame(d[f"kin_{a}"], columns=cols)
           for a in d["animals"]}
    return kin, d["logit_coef"].astype(float), float(d["logit_intercept"][0])


def phase_zstats(k: pd.DataFrame, mouse: str, phase: str):
    t = k["t"].to_numpy()
    m = t < A.PARTNER_INTRO_S if phase == "alone" else t >= A.PARTNER_INTRO_S
    out = {}
    for f in KIN_FEATS:
        v = k[f"{mouse}_{f}"].to_numpy()[m]
        out[f] = (np.nanmean(v), np.nanstd(v))
    return out


def kin_score(k: pd.DataFrame, mouse: str, times: np.ndarray, phase: str,
              coef: np.ndarray, intercept: float) -> np.ndarray:
    """Logistic emission score for one animal's kinematics at given times."""
    X = kin_feature_matrix(k, mouse, times, phase)
    s = X @ coef + intercept
    s[~np.isfinite(X).all(axis=1)] = np.nan
    return s


# ------------------------------------------------------------------ voice
def voice_logodds(sessions: list[CORE.Session], seed: int = 0) -> dict:
    """EM fit (no stim tying assumed: stim_key == session key) -> per-call
    log-likelihood ratio llR - llP (prior-free, for fusion)."""
    res = CORE.fit_attribution(sessions, imodel=None, seed=seed)
    g = res["_globals"]
    out = {}
    for s in sessions:
        if not len(s.X):
            out[s.key] = np.zeros(0)
            continue
        llR = CORE._gauss_logpdf_diag(s.X, res[s.key]["muR"], g["varR"])
        llP = CORE._gauss_logpdf_diag(s.X, res[s.key]["muP"], g["varP"])
        out[s.key] = llR - llP
    return out


# ------------------------------------------------------------------ level
class LevelCue:
    def __init__(self, enabled, ridge=None, cols=None, mic=None, b=0.0,
                 sigma=3.0, r2_loao=np.nan):
        self.enabled = enabled
        self.ridge, self.cols, self.mic, self.b = ridge, cols, mic, b
        self.sigma, self.r2_loao = sigma, r2_loao

    @classmethod
    def fit(cls, alone: pd.DataFrame, min_r2: float = 0.05):
        cols = ["rf_dur_ms", "rf_tonality", "rf_f_mean_khz", "rf_spread_khz",
                "rf_entropy_bits", "rf_amp_cv"]
        ok = (np.isfinite(alone[cols]).all(axis=1)
              & np.isfinite(alone["rf_amp_peak_db"])
              & np.isfinite(alone[["m1_x", "m1_y"]]).all(axis=1))
        d = alone[ok]
        Xc = d[cols].to_numpy(float)
        L = d["rf_amp_peak_db"].to_numpy(float)
        P = d[["m1_x", "m1_y"]].to_numpy(float)
        an = d["animal_id"].to_numpy()
        ridge = Ridge(alpha=1.0).fit(Xc, L)
        resid = L - ridge.predict(Xc)

        def fit_dist(res, pos):
            def f(th):
                mx, my, b, a0 = th
                dd = np.hypot(pos[:, 0] - mx, pos[:, 1] - my)
                return (a0 - 10 * b * np.log10(np.maximum(dd, 10))) - res
            return least_squares(f, [360, 700, 1.0, np.median(res)],
                                 loss="soft_l1", f_scale=3.0)

        full = fit_dist(resid, P)
        mx, my, b, a0 = full.x
        pred = a0 - 10 * b * np.log10(np.maximum(
            np.hypot(P[:, 0] - mx, P[:, 1] - my), 10))
        ss_tot = np.sum((resid - resid.mean()) ** 2) + 1e-12
        press = 0.0
        for a in np.unique(an):
            tr = an != a
            if tr.sum() < 20 or (~tr).sum() < 2:
                press += np.sum((resid[~tr] - resid[tr].mean()) ** 2)
                continue
            f2 = fit_dist(resid[tr], P[tr])
            mx2, my2, b2, a02 = f2.x
            dte = np.hypot(P[~tr, 0] - mx2, P[~tr, 1] - my2)
            press += np.sum((resid[~tr] - (a02 - 10 * b2 * np.log10(
                np.maximum(dte, 10)))) ** 2)
        r2_loao = 1 - press / ss_tot
        sigma = float(np.std(resid - pred, ddof=4))
        return cls(bool(r2_loao > min_r2), ridge, cols,
                   np.array([mx, my]), float(b), max(sigma, 1.0), float(r2_loao))

    def logodds(self, feats: pd.DataFrame, pos_R: np.ndarray,
                pos_P: np.ndarray) -> np.ndarray:
        n = len(feats)
        if not self.enabled:
            return np.zeros(n)
        Xc = feats[self.cols].to_numpy(float)
        L = feats["rf_amp_peak_db"].to_numpy(float)
        base = self.ridge.predict(np.where(np.isfinite(Xc), Xc, 0.0))
        resid = L - base
        out = np.zeros(n)
        for sign, pos in ((1.0, pos_R), (-1.0, pos_P)):
            dd = np.hypot(pos[:, 0] - self.mic[0], pos[:, 1] - self.mic[1])
            mu = -10 * self.b * np.log10(np.maximum(dd, 10))
            z = (resid - (mu + np.nanmedian(resid - mu))) / self.sigma
            ll = -0.5 * z ** 2
            ll[~np.isfinite(ll)] = 0.0
            out += sign * ll
        both = np.isfinite(pos_R).all(axis=1) & np.isfinite(pos_P).all(axis=1)
        out[~both] = 0.0
        return out


# ------------------------------------------------------------------ overlap
def overlap_scores(crops: np.ndarray, thr: float = 0.45) -> np.ndarray:
    """Per call: fraction of active time-columns with two simultaneous
    non-harmonic frequency bands. crops: (n, 64, 64), row 0 = 40 kHz."""
    f_of_row = 40_000 + (125_000 - 40_000) * np.arange(64) / 63.0
    out = np.zeros(len(crops))
    for i, c in enumerate(crops):
        act = c > thr
        n_double, n_active = 0, 0
        for col in range(64):
            rows = np.where(act[:, col])[0]
            if len(rows) < 2:
                continue
            n_active += 1
            splits = np.where(np.diff(rows) > 8)[0]        # gap > ~10 kHz
            if not len(splits):
                continue
            runs = np.split(rows, splits + 1)
            runs = [r for r in runs if len(r) >= 2]
            if len(runs) < 2:
                continue
            f1, f2 = sorted(f_of_row[int(np.mean(r))] for r in runs[:2])
            if abs(f2 / f1 - 2.0) > 0.15:                  # not a harmonic
                n_double += 1
        out[i] = n_double / max(n_active, 1)
    return out


# ------------------------------------------------------------------ fusion
class Fusion:
    """Logistic fusion of cue log-odds -> calibrated posterior."""

    def __init__(self, cue_names: list[str]):
        self.cue_names = cue_names
        self.clf = None

    @staticmethod
    def _prep(M: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(M), M, 0.0)

    def fit(self, M: np.ndarray, y: np.ndarray):
        # balanced: fused scores are prior-free log-odds; session priors are
        # handled downstream (HMM), never baked in from benchmark base rates
        self.clf = LogisticRegression(max_iter=1000, C=1.0,
                                      class_weight="balanced").fit(self._prep(M), y)
        return self

    def logodds(self, M: np.ndarray) -> np.ndarray:
        return self.clf.decision_function(self._prep(M))

    @property
    def weights(self):
        return dict(zip(self.cue_names, self.clf.coef_[0].round(3).tolist()),
                    intercept=round(float(self.clf.intercept_[0]), 3))


def hmm_posterior(times: np.ndarray, logodds: np.ndarray,
                  pi: float = 0.5) -> np.ndarray:
    return CORE.hmm_smooth(times, logodds / 2.0, -logodds / 2.0, pi)
