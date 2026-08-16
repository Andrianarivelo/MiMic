"""Core machinery for v5: directionality, effects and the HET deficit.

The central statistic is the ANTISYMMETRIC CROSS-CORRELOGRAM.

Let x(t) be a call train and y(t) an event train (e.g. onsets of a partner
behaviour), both on a 100 ms grid and mean-centred within a session.  Write

    R(L) = (1/T) sum_t x(t) y(t + L)

and split it into its even and odd parts

    R_even(L) = [R(L) + R(-L)] / 2       R_odd(L) = [R(L) - R(-L)] / 2 .

Claim: an instantaneous common drive contributes ONLY to R_even.  If
x(t) = a z(t) + e_x(t) and y(t) = b z(t) + e_y(t) with the noises independent
of z and of each other, then R(L) = a b R_z(L), and the autocovariance R_z of
any weakly stationary process satisfies R_z(L) = R_z(-L).  Hence R_odd == 0.

So R_odd is a directional statistic that is *blind* to shared slow modulation:
proximity, arousal, bout structure -- and the regression-to-the-mean artefact
that sank the v4 consequence claim -- all enter symmetrically and cancel.

Sign convention: R(L) peaks at L = +l when x leads y by l.  Therefore
R_odd(L) > 0 for L > 0 means the CALL LEADS the behaviour; R_odd(L) < 0 for
L > 0 means the BEHAVIOUR LEADS the call.

Null: circular shifts of x.  A circular shift preserves x's own
autocorrelation and bout structure exactly and leaves y untouched, so it tests
the cross-relation and nothing else.  With T bins there are T distinct shifts,
and all of them can be evaluated at once by FFT, giving an *exact* permutation
distribution per session rather than a sampled one.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import attr_common as A

BIN_S = 0.1
T0, T1 = 300.0, 900.0          # interaction window
NBIN = int(round((T1 - T0) / BIN_S))   # 6000
FLAGS = [
    "nose2nose", "sidebyside", "sidereside", "nose2anogenital", "nose2body",
    "oriented_toward", "following", "chasing", "approach",
    "withdrawal_from_partner", "escape", "withdrawal_after_contact",
    "fighting", "rearing", "passive",
]
# Flags that mark the ACTOR (the animal performing the act).  Mutual flags are
# shared between the two animals and cannot support a directional read.
DIRECTIONAL = {
    "nose2anogenital", "nose2body", "oriented_toward", "following", "chasing",
    "approach", "withdrawal_from_partner", "escape",
    "withdrawal_after_contact", "rearing",
}
MUTUAL = {"nose2nose", "sidebyside", "sidereside", "fighting", "passive"}

PRETTY = {
    "nose2nose": "nose-to-nose", "sidebyside": "side-by-side",
    "sidereside": "side-reverse-side", "nose2anogenital": "anogenital sniff",
    "nose2body": "body sniff", "oriented_toward": "oriented toward",
    "following": "following", "chasing": "chasing", "approach": "approach",
    "withdrawal_from_partner": "withdrawal", "escape": "escape",
    "withdrawal_after_contact": "withdrawal after contact",
    "fighting": "fighting", "rearing": "rearing", "passive": "passive",
}

MIN_OFF_BINS = 5      # a bout onset needs >=0.5 s of the flag being off
RNG = np.random.default_rng(20260815)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_bins() -> pd.DataFrame:
    out = A.ensure_out()
    b = pd.read_csv(out / "beh_bins.csv.gz", low_memory=False)
    return b[b["t"] >= T0].reset_index(drop=True)


def load_calls_ctx() -> pd.DataFrame:
    """Per-call table: acoustics (our own ridge features) + GEO attribution."""
    out = A.ensure_out()
    cf = pd.read_csv(out / "call_features.csv", low_memory=False)
    geo = pd.read_csv(out / "v2_geo_attribution.csv", low_memory=False)
    geo = geo[["call_id", "p_res_hmm", "geo_logodds"]]
    df = cf.merge(geo, on="call_id", how="left")
    df = df[(df["start_s"] >= T0) & (df["start_s"] < T1)].copy()
    df["genotype"] = df["animal_id"].astype(str).map(A.GENOTYPE_MAP)
    df["p_res"] = df["p_res_hmm"].fillna(A.__dict__.get("PI_WT", 0.916))
    return df.reset_index(drop=True)


def onset_train(flag: np.ndarray, min_off: int = MIN_OFF_BINS) -> np.ndarray:
    """1 at bins where `flag` turns on after >= min_off bins of being off."""
    f = (np.asarray(flag) > 0.5).astype(np.int8)
    on = np.zeros_like(f)
    if f.size == 0:
        return on.astype(float)
    prev_off = 0
    for i in range(f.size):
        if f[i]:
            if prev_off >= min_off:
                on[i] = 1
            prev_off = 0
        else:
            prev_off += 1
    return on.astype(float)


def session_arrays(bins: pd.DataFrame, calls: pd.DataFrame) -> dict:
    """Per-session dict of aligned 100 ms trains."""
    sess = {}
    for aid, g in bins.groupby("animal_id"):
        g = g.sort_values("t")
        n = len(g)
        if n < NBIN * 0.9:
            continue
        d = {"genotype": g["genotype"].iloc[0], "n": n,
             "t": g["t"].to_numpy()}
        for m in ("m1", "m2"):
            for f in FLAGS:
                v = g[f"{m}_{f}"].to_numpy(float)
                d[f"{m}_{f}"] = v
                d[f"on_{m}_{f}"] = onset_train(v)
        for k in ("speed_m1", "speed_m2", "dist", "app_m1", "app_m2"):
            d[k] = g[k].to_numpy(float)
        # call trains built from the per-call table (weighted by attribution)
        c = calls[calls["animal_id"].astype(str) == str(aid)]
        idx = np.clip(((c["start_s"].to_numpy() - T0) / BIN_S).astype(int), 0, n - 1)
        x_all = np.zeros(n)
        x_res = np.zeros(n)
        np.add.at(x_all, idx, 1.0)
        np.add.at(x_res, idx, c["p_res"].to_numpy())
        d["x_all"] = x_all
        d["x_res"] = x_res
        d["call_idx"] = idx
        d["calls"] = c
        sess[str(aid)] = d
    return sess


# --------------------------------------------------------------------------
# antisymmetric cross-correlogram
# --------------------------------------------------------------------------
def xcov_all_shifts(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """c[L] = sum_t x(t) y(t+L) for every circular lag L (length n)."""
    n = x.size
    return np.fft.irfft(np.conj(np.fft.rfft(x, n)) * np.fft.rfft(y, n), n)


def odd_profile(x: np.ndarray, y: np.ndarray, lmax: int):
    """Return (R(+L), R(-L)) arrays for L = 1..lmax, unnormalised counts."""
    c = xcov_all_shifts(x, y)
    pos = c[1:lmax + 1]
    neg = c[-lmax:][::-1]      # c[-L] for L = 1..lmax
    return pos, neg


def exact_shift_null(x: np.ndarray, y: np.ndarray, lmax: int):
    """Every circular shift of x, evaluated at once.

    Returns arrays (n_shifts, lmax) of R(+L) and R(-L) under the null, where
    shift s corresponds to x rolled by s.  Because rolling x by s maps
    R(L) -> R(L - s), the full null is just the same cross-covariance vector
    read at different offsets -- no resampling required.
    """
    c = xcov_all_shifts(x, y)
    n = c.size
    L = np.arange(1, lmax + 1)
    S = np.arange(n)[:, None]
    pos = c[(L[None, :] - S) % n]
    neg = c[(-L[None, :] - S) % n]
    return pos, neg


def pooled_directionality(sess: dict, xkey: str, ykey: str, lmax: int,
                          keep: list[str] | None = None, n_draw: int = 4000):
    """Pool the odd cross-correlogram over sessions with a joint shift null.

    Returns dict with the observed peri-event call-rate profile (calls/min at
    each lag, and the same for the null mean), the odd component, the summary
    statistic Lambda = sum_{L>0} R_odd(L) and its two-sided p.
    """
    keep = keep or list(sess)
    pos_o = np.zeros(lmax)
    neg_o = np.zeros(lmax)
    n_ev = 0.0
    n_call = 0.0
    per_sess = {}
    nulls_pos, nulls_neg = [], []
    for aid in keep:
        d = sess[aid]
        x = d[xkey]
        y = d[ykey]
        if x.sum() < 1 or y.sum() < 1:
            continue
        p, q = odd_profile(x, y, lmax)
        pos_o += p
        neg_o += q
        n_ev += y.sum()
        n_call += x.sum()
        per_sess[aid] = (p, q, y.sum())
        np_, nn_ = exact_shift_null(x, y, lmax)
        sel = RNG.integers(0, np_.shape[0], size=n_draw)
        nulls_pos.append(np_[sel])
        nulls_neg.append(nn_[sel])
    if not per_sess:
        return None
    NP = np.sum(nulls_pos, axis=0)     # (n_draw, lmax)
    NN = np.sum(nulls_neg, axis=0)
    odd_obs = 0.5 * (pos_o - neg_o)
    odd_null = 0.5 * (NP - NN)
    lam_obs = odd_obs.sum()
    lam_null = odd_null.sum(axis=1)
    p_two = (np.sum(np.abs(lam_null) >= abs(lam_obs)) + 1) / (len(lam_null) + 1)
    # rate profile in calls/min per event
    rate_pos = pos_o / max(n_ev, 1) / BIN_S * 60.0
    rate_neg = neg_o / max(n_ev, 1) / BIN_S * 60.0
    rate_null = NP.mean(axis=0) / max(n_ev, 1) / BIN_S * 60.0
    return {
        "lags_s": np.arange(1, lmax + 1) * BIN_S,
        "rate_pos": rate_pos, "rate_neg": rate_neg, "rate_null": rate_null,
        "odd_obs": odd_obs,
        "odd_lo": np.percentile(odd_null, 2.5, axis=0),
        "odd_hi": np.percentile(odd_null, 97.5, axis=0),
        "lambda": float(lam_obs),
        "lambda_z": float((lam_obs - lam_null.mean()) / (lam_null.std() + 1e-12)),
        "p": float(p_two),
        "n_events": float(n_ev), "n_calls": float(n_call),
        "n_sessions": len(per_sess),
        "per_session": {k: (v[0] - v[1]).sum() for k, v in per_sess.items()},
    }


def bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


def cluster_boot(vals_by_session: dict, stat, n_boot: int = 4000):
    """Bootstrap over sessions; `stat` maps a list of per-session items -> float."""
    keys = list(vals_by_session)
    obs = stat([vals_by_session[k] for k in keys])
    draws = np.empty(n_boot)
    for b in range(n_boot):
        sel = RNG.integers(0, len(keys), len(keys))
        draws[b] = stat([vals_by_session[keys[i]] for i in sel])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return obs, lo, hi, draws
