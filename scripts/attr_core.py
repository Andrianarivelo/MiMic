"""Core model for single-microphone call attribution.

A call in the partner phase was emitted by the resident (R) or the partner (P).
Three independent information sources are combined per call:

1. VOICE: individual acoustic fingerprint in a PCA voice space built from
   handcrafted ridge features + from-scratch autoencoder latents. The resident
   voice is anchored by that animal's alone-phase calls (only it could call);
   the partner voice is a free component TIED ACROSS SESSIONS that used the
   same stim animal.
2. INTENSITY x POSITION: received level depends on the caller's distance to
   the microphone. The attenuation model (mic position, exponent, per-animal
   source level) is fitted on alone-phase calls where the caller position is
   known unambiguously, then evaluated for both candidate positions.
3. BOUT CONTINUITY: USVs come in bouts; consecutive calls with short gaps
   rarely switch caller. A 2-state HMM with gap-dependent switching smooths
   per-call posteriors along the session.

The same fit/predict machinery is reused by the benchmarks on synthetic
"pseudo-sessions" with ground-truth speakers.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Handcrafted features that describe call SHAPE (identity-relevant), not level.
VOICE_FEATS = [
    "rf_dur_ms", "rf_f_mean_khz", "rf_f_std_khz", "rf_f_min_khz", "rf_f_max_khz",
    "rf_f_start_khz", "rf_f_end_khz", "rf_slope_khz_ms", "rf_abs_slope_khz_ms",
    "rf_curve_khz_ms2", "rf_sinuosity", "rf_jumps", "rf_frac_rising",
    "rf_entropy_bits", "rf_spread_khz", "rf_amp_cv", "rf_amp_peak_pos",
    "rf_tonality",
]

PCA_DIMS = 12
KAPPA_ANCHOR = 3.0        # pseudo-count shrinking resident mean to population
CONTEXT_VAR_INFLATE = 2.0  # admit alone->social repertoire drift of the resident
EM_ITERS = 60
PI_PSEUDO = 2.0           # Beta pseudo-counts on the session mixing weight
HMM_P_MIN = 0.02          # floor switching probability between consecutive calls
HMM_TAU_S = 2.0           # gap constant for caller switching


# --------------------------------------------------------------------------
def build_voice_space(feats: pd.DataFrame, latents: np.ndarray | None,
                      n_pca: int = PCA_DIMS):
    """Fit scaler+PCA on all calls; return (X, transform_fn)."""
    H = feats[VOICE_FEATS].to_numpy(dtype=float)
    H = np.where(np.isfinite(H), H, np.nan)
    col_med = np.nanmedian(H, axis=0)
    H = np.where(np.isnan(H), col_med[None, :], H)
    blocks = [H]
    if latents is not None:
        blocks.append(latents.astype(float))
    M = np.concatenate(blocks, axis=1)
    scaler = StandardScaler().fit(M)
    Ms = scaler.transform(M)
    pca = PCA(n_components=min(n_pca, Ms.shape[1]), random_state=0).fit(Ms)
    X = pca.transform(Ms)

    def transform(f2: pd.DataFrame, l2: np.ndarray | None):
        H2 = f2[VOICE_FEATS].to_numpy(dtype=float)
        H2 = np.where(np.isfinite(H2), H2, col_med[None, :])
        b = [H2] + ([l2.astype(float)] if l2 is not None else [])
        return pca.transform(scaler.transform(np.concatenate(b, axis=1)))

    return X, transform, pca


# --------------------------------------------------------------------------
@dataclasses.dataclass
class IntensityModel:
    mic: np.ndarray          # (2,) arena px
    b: float                 # attenuation exponent scale
    a0: float                # population source level (dB)
    sigma: float             # residual sd (dB)
    r2_insample: float
    r2_loao: float           # leave-one-animal-out R^2 (predicting with a0)
    enabled: bool

    def expected(self, pos_xy: np.ndarray) -> np.ndarray:
        d = np.hypot(pos_xy[:, 0] - self.mic[0], pos_xy[:, 1] - self.mic[1])
        return self.a0 - 10.0 * self.b * np.log10(np.maximum(d, 10.0))


def fit_intensity_model(level_db: np.ndarray, pos_xy: np.ndarray,
                        animal_idx: np.ndarray, arena_hint=(720, 720),
                        min_r2: float = 0.05) -> IntensityModel:
    """Fit L = a_animal - 10 b log10(dist(pos, mic)) on alone-phase calls."""
    ok = np.isfinite(level_db) & np.isfinite(pos_xy).all(axis=1)
    L, Pxy, ai = level_db[ok], pos_xy[ok], animal_idx[ok]
    animals = np.unique(ai)
    n_an = len(animals)
    a_map = {a: k for k, a in enumerate(animals)}
    ai_k = np.array([a_map[a] for a in ai])

    def resid(theta):
        mx, my, b = theta[:3]
        a0 = theta[3]
        ua = theta[4:]
        d = np.hypot(Pxy[:, 0] - mx, Pxy[:, 1] - my)
        pred = (a0 + ua[ai_k]) - 10.0 * b * np.log10(np.maximum(d, 10.0))
        return np.concatenate([pred - L, 3.0 * ua])  # ridge on animal offsets

    x0 = np.concatenate([[arena_hint[0] / 2, arena_hint[1] / 2, 1.0, np.median(L)],
                         np.zeros(n_an)])
    fit = least_squares(resid, x0, loss="soft_l1", f_scale=3.0)
    mx, my, b, a0 = fit.x[:4]
    ua = fit.x[4:]
    d = np.hypot(Pxy[:, 0] - mx, Pxy[:, 1] - my)
    pred = (a0 + ua[ai_k]) - 10.0 * b * np.log10(np.maximum(d, 10.0))
    ss_res = np.sum((L - pred) ** 2)
    ss_tot = np.sum((L - L.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    # leave-one-animal-out: refit without each animal, predict its calls with a0
    press = 0.0
    for a in animals:
        tr = ai != a
        te = ~tr
        if te.sum() < 2 or tr.sum() < 20:
            press += np.sum((L[te] - L[tr].mean()) ** 2)
            continue
        Ltr, Ptr, aitr = L[tr], Pxy[tr], ai[tr]
        atr = np.unique(aitr)
        amap2 = {x: k for k, x in enumerate(atr)}
        ak2 = np.array([amap2[x] for x in aitr])

        def r2fun(theta, Ltr=Ltr, Ptr=Ptr, ak2=ak2, n2=len(atr)):
            mx2, my2, b2, a02 = theta[:4]
            ua2 = theta[4:]
            dd = np.hypot(Ptr[:, 0] - mx2, Ptr[:, 1] - my2)
            pr = (a02 + ua2[ak2]) - 10.0 * b2 * np.log10(np.maximum(dd, 10.0))
            return np.concatenate([pr - Ltr, 3.0 * ua2])

        x02 = np.concatenate([[arena_hint[0] / 2, arena_hint[1] / 2, 1.0,
                               np.median(Ltr)], np.zeros(len(atr))])
        f2 = least_squares(r2fun, x02, loss="soft_l1", f_scale=3.0)
        mx2, my2, b2, a02 = f2.x[:4]
        dte = np.hypot(Pxy[te, 0] - mx2, Pxy[te, 1] - my2)
        prte = a02 - 10.0 * b2 * np.log10(np.maximum(dte, 10.0))
        press += np.sum((L[te] - prte) ** 2)
    r2_loao = 1.0 - press / ss_tot

    sigma = float(np.std(L - pred, ddof=4))
    return IntensityModel(
        mic=np.array([mx, my]), b=float(b), a0=float(a0), sigma=max(sigma, 1.0),
        r2_insample=float(r2), r2_loao=float(r2_loao),
        enabled=bool(r2_loao > min_r2),
    )


# --------------------------------------------------------------------------
@dataclasses.dataclass
class Session:
    """One (pseudo-)session to attribute. Arrays are per partner-phase call."""
    key: str
    stim_key: str            # sessions sharing this key share the partner voice
    X: np.ndarray            # (n, d) voice-space features
    times: np.ndarray        # (n,) call times for the HMM
    level_db: np.ndarray     # (n,) received level (NaN allowed)
    pos_R: np.ndarray        # (n, 2) resident position (NaN allowed)
    pos_P: np.ndarray        # (n, 2) partner position (NaN allowed)
    anchor_X: np.ndarray     # (m, d) resident's alone-phase calls (m may be 0)


def _gauss_logpdf_diag(X, mu, var):
    return -0.5 * (np.sum((X - mu) ** 2 / var, axis=1)
                   + np.sum(np.log(2 * np.pi * var)))


def _intensity_loglik(sess: Session, imodel: IntensityModel | None):
    """Per-call intensity log-likelihood for caller R and P (0 when unusable)."""
    n = len(sess.X)
    llR = np.zeros(n)
    llP = np.zeros(n)
    if imodel is None or not imodel.enabled:
        return llR, llP
    for arr, pos in ((llR, sess.pos_R), (llP, sess.pos_P)):
        ok = np.isfinite(sess.level_db) & np.isfinite(pos).all(axis=1)
        mu = imodel.expected(pos[ok])
        z = (sess.level_db[ok] - mu) / imodel.sigma
        arr[ok] = -0.5 * z ** 2 - np.log(imodel.sigma)
    both = (np.isfinite(sess.pos_R).all(axis=1) & np.isfinite(sess.pos_P).all(axis=1)
            & np.isfinite(sess.level_db))
    llR[~both] = 0.0
    llP[~both] = 0.0
    return llR, llP


def hmm_smooth(times: np.ndarray, ll_R: np.ndarray, ll_P: np.ndarray,
               pi: float, p_min: float = HMM_P_MIN, tau: float = HMM_TAU_S):
    """Forward-backward over the call sequence; returns P(caller=R) per call."""
    n = len(times)
    if n == 0:
        return np.zeros(0)
    order = np.argsort(times)
    lR, lP = ll_R[order], ll_P[order]
    gaps = np.diff(times[order])
    p_sw = p_min + (0.5 - p_min) * (1.0 - np.exp(-np.maximum(gaps, 0) / tau))

    la = np.zeros((n, 2))
    la[0] = [np.log(pi + 1e-12) + lR[0], np.log(1 - pi + 1e-12) + lP[0]]
    for i in range(1, n):
        lt_stay = np.log(1 - p_sw[i - 1])
        lt_sw = np.log(p_sw[i - 1])
        la[i, 0] = np.logaddexp(la[i - 1, 0] + lt_stay, la[i - 1, 1] + lt_sw) + lR[i]
        la[i, 1] = np.logaddexp(la[i - 1, 1] + lt_stay, la[i - 1, 0] + lt_sw) + lP[i]
    lb = np.zeros((n, 2))
    for i in range(n - 2, -1, -1):
        lt_stay = np.log(1 - p_sw[i])
        lt_sw = np.log(p_sw[i])
        lb[i, 0] = np.logaddexp(lb[i + 1, 0] + lt_stay + lR[i + 1],
                                lb[i + 1, 1] + lt_sw + lP[i + 1])
        lb[i, 1] = np.logaddexp(lb[i + 1, 1] + lt_stay + lP[i + 1],
                                lb[i + 1, 0] + lt_sw + lR[i + 1])
    lg = la + lb
    p = 1.0 / (1.0 + np.exp(np.clip(lg[:, 1] - lg[:, 0], -500, 500)))
    out = np.empty(n)
    out[order] = p
    return out


def fit_attribution(sessions: list[Session], imodel: IntensityModel | None,
                    em_iters: int = EM_ITERS, seed: int = 0):
    """Hierarchical two-source EM over all sessions with partner-voice tying.

    Returns dict per session key with p_res_voice / p_res_full / p_res_hmm and
    the fitted component means.
    """
    rng = np.random.default_rng(seed)
    d = sessions[0].X.shape[1]

    all_X = np.concatenate([s.X for s in sessions], axis=0)
    all_anchor = np.concatenate(
        [s.anchor_X for s in sessions if len(s.anchor_X)], axis=0)
    pop_alone_mu = (all_anchor.mean(axis=0) if len(all_anchor) else
                    all_X.mean(axis=0))
    # within-voice variance: pooled within-animal scatter of anchors; fallback
    # to a fraction of the global scatter
    wc = []
    for s in sessions:
        if len(s.anchor_X) >= 3:
            wc.append(s.anchor_X - s.anchor_X.mean(axis=0))
    var_within = (np.concatenate(wc).var(axis=0) if wc else all_X.var(axis=0) * 0.5)
    var_within = np.maximum(var_within, 1e-3)
    varR = var_within * CONTEXT_VAR_INFLATE
    varP = var_within * CONTEXT_VAR_INFLATE

    # resident means: anchored + shrunk
    muR = {}
    for s in sessions:
        m = len(s.anchor_X)
        xbar = s.anchor_X.mean(axis=0) if m else pop_alone_mu
        muR[s.key] = (m * xbar + KAPPA_ANCHOR * pop_alone_mu) / (m + KAPPA_ANCHOR)

    # partner means: tied by stim_key; init from calls far from muR
    stim_keys = sorted({s.stim_key for s in sessions})
    muP = {}
    for q in stim_keys:
        pool = []
        for s in sessions:
            if s.stim_key != q or not len(s.X):
                continue
            dist = np.sum((s.X - muR[s.key]) ** 2 / varR, axis=1)
            k = max(1, int(0.3 * len(dist)))
            pool.append(s.X[np.argsort(dist)[-k:]])
        muP[q] = (np.concatenate(pool).mean(axis=0) if pool
                  else pop_alone_mu + rng.normal(0, 0.1, d))

    pi = {s.key: 0.7 for s in sessions}

    for it in range(em_iters):
        # E-step
        resp = {}
        for s in sessions:
            if not len(s.X):
                resp[s.key] = np.zeros(0)
                continue
            llR = _gauss_logpdf_diag(s.X, muR[s.key], varR)
            llP = _gauss_logpdf_diag(s.X, muP[s.stim_key], varP)
            iR, iP = _intensity_loglik(s, imodel)
            a = np.log(pi[s.key]) + llR + iR
            b = np.log(1 - pi[s.key]) + llP + iP
            resp[s.key] = 1.0 / (1.0 + np.exp(np.clip(b - a, -500, 500)))
        # M-step
        for s in sessions:
            r = resp[s.key]
            m = len(s.anchor_X)
            wa = KAPPA_ANCHOR + m
            xa = (s.anchor_X.sum(axis=0) + KAPPA_ANCHOR * pop_alone_mu)
            if len(r):
                muR[s.key] = (xa + (r[:, None] * s.X).sum(axis=0)) / (wa + r.sum())
            pi_new = (r.sum() + PI_PSEUDO) / (len(r) + 2 * PI_PSEUDO) if len(r) else 0.7
            pi[s.key] = float(np.clip(pi_new, 0.02, 0.98))
        for q in stim_keys:
            num = np.zeros(d)
            den = 1.0  # prior pseudo-count toward population mean
            numv = pop_alone_mu.copy()
            for s in sessions:
                if s.stim_key != q or not len(s.X):
                    continue
                w = 1.0 - resp[s.key]
                num += (w[:, None] * s.X).sum(axis=0)
                den += w.sum()
            muP[q] = (numv + num) / den

    # final per-call outputs
    out = {}
    for s in sessions:
        n = len(s.X)
        if not n:
            out[s.key] = dict(p_res_voice=np.zeros(0), p_res_full=np.zeros(0),
                              p_res_hmm=np.zeros(0), muR=muR[s.key],
                              muP=muP[s.stim_key], pi=pi[s.key])
            continue
        llR = _gauss_logpdf_diag(s.X, muR[s.key], varR)
        llP = _gauss_logpdf_diag(s.X, muP[s.stim_key], varP)
        iR, iP = _intensity_loglik(s, imodel)
        pv = 1 / (1 + np.exp(np.clip((np.log(1 - pi[s.key]) + llP)
                                     - (np.log(pi[s.key]) + llR), -500, 500)))
        pf = 1 / (1 + np.exp(np.clip((np.log(1 - pi[s.key]) + llP + iP)
                                     - (np.log(pi[s.key]) + llR + iR), -500, 500)))
        ph = hmm_smooth(s.times, llR + iR, llP + iP, pi[s.key])
        out[s.key] = dict(p_res_voice=pv, p_res_full=pf, p_res_hmm=ph,
                          muR=muR[s.key], muP=muP[s.stim_key], pi=pi[s.key])
    out["_globals"] = dict(varR=varR, varP=varP, pop_alone_mu=pop_alone_mu)
    return out
