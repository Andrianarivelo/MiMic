"""Call x behaviour analysis with movement-adjusted enrichment and shift nulls.

A  ENRICHMENT   call rate inside vs outside each behaviour, raw and adjusted
                for speed x distance strata (Mantel-Haenszel person-time rate
                ratio); significance from circular-shift nulls; BH-FDR.
B  PERI-EVENT   behaviour probability around calls, and call rate around
                behaviour onsets, both against shift nulls.
C  CONSEQUENCE  matched-counterfactual: for each call, control bins in the same
                session matched on behaviour state, speed quintile and distance
                tercile; compare the following 2 s of own/partner kinematics
                and behaviour.
D  PREDICTION   LOSO-CV models of per-bin call occurrence: kinematics only,
                behaviour only, both.
E  CALL TYPES   behavioural profile per syllable class and per duration
                tercile, with a permutation test for heterogeneity.

All shift nulls use the exact circular cross-correlation, so the null is taken
over every possible time shift rather than a random sample of them. Shifting
call times preserves call bout structure and behaviour structure exactly.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A
from attr4_00_build_behavior import BEH, BIN_S

PARTNER_T0 = 305.0
N_PERM = 2000
PERI_HALF_S = 5.0
SEED = A.RNG_SEED
KIN_COLS = ["speed_m1", "speed_m2", "dist", "app_m1", "app_m2"]
N_STRATA = 9
# a-priori sets, fixed before looking at the results
ACTIVE_SET = ("nose2anogenital", "nose2body", "following", "chasing", "approach")
DIRECTIONAL = ("nose2anogenital", "nose2body", "oriented_toward", "following",
               "chasing", "approach")


def bh_fdr(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    ps = p[ok]
    n = len(ps)
    order = np.argsort(ps)
    ranked = ps[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def xcorr(a, b):
    """Circular cross-correlation: out[L] = sum_i a[i] * b[(i+L) mod n]."""
    n = len(a)
    return np.fft.irfft(np.conj(np.fft.rfft(a)) * np.fft.rfft(b), n)


# ---------------------------------------------------------------- loading
def load_bins():
    df = pd.read_csv(A.OUT_DIR / "beh_bins.csv.gz", compression="gzip",
                     low_memory=False)
    df["animal_id"] = df["animal_id"].astype(str)
    return df


def session_arrays(df):
    S = {}
    flag_names = [f"m1_{b}" for b in BEH] + [f"m2_{b}" for b in BEH]
    for animal, g in df[df["t"] >= PARTNER_T0].groupby("animal_id"):
        g = g.sort_values("bin").reset_index(drop=True)
        counts = g["n_calls"].to_numpy().astype(float)
        if counts.sum() < 1:
            continue
        flags = (g[flag_names].to_numpy() > 0)

        def terc(v, qs):
            v = np.where(np.isfinite(v), v, np.nanmedian(v))
            return np.digitize(v, np.nanquantile(v, qs))
        strat = (terc(g["speed_m1"].to_numpy(), [1 / 3, 2 / 3]) * 3
                 + terc(g["dist"].to_numpy(), [1 / 3, 2 / 3]))
        # a-priori composites: one well-powered test instead of many weak ones
        comp, comp_names = [], []
        for tag in ("m1", "m2"):
            act = np.zeros(len(counts), bool)
            for b in ACTIVE_SET:
                act |= g[f"{tag}_{b}"].to_numpy() > 0
            comp.append(act)
            comp_names.append(f"{tag}_ACTIVEsocial")
        cont = np.zeros(len(counts), bool)
        for b in ("nose2nose", "sidebyside", "sidereside"):
            cont |= g[f"m1_{b}"].to_numpy() > 0
        comp.append(cont)
        comp_names.append("m1_CONTACTmutual")
        flags = np.hstack([flags, np.stack(comp, axis=1)])
        S[animal] = dict(counts=counts, flags=flags, strat=strat, g=g,
                         names=flag_names + comp_names, n=len(counts))
    return S


def actor_asymmetry(S, P, rng, behaviors=DIRECTIONAL):
    """Single test of 'calls track the ACTOR': mean over directional
    behaviours of log2RR(resident is actor) - log2RR(partner is actor),
    against the same circular-shift null."""
    names = S[next(iter(S))]["names"]
    pairs = [(names.index(f"m1_{b}"), names.index(f"m2_{b}"))
             for b in behaviors
             if f"m1_{b}" in names and f"m2_{b}" in names]
    animals = list(S)

    def stat(shifts):
        d = []
        for j1, j2 in pairs:
            r1 = pooled_rr(P, animals, j1, shifts, True)
            r2 = pooled_rr(P, animals, j2, shifts, True)
            if np.isfinite(r1) and np.isfinite(r2) and r1 > 0 and r2 > 0:
                d.append(np.log2(r1) - np.log2(r2))
        return np.nanmean(d) if d else np.nan

    obs = stat([0] * len(animals))
    null = np.array([stat([int(rng.integers(1, P[a]["n"])) for a in animals])
                     for _ in range(N_PERM)])
    v = null[np.isfinite(null)]
    p = (np.sum(np.abs(v - v.mean()) >= abs(obs - v.mean())) + 1) / (len(v) + 1)
    per = {}
    for b, (j1, j2) in zip(behaviors, pairs):
        r1 = pooled_rr(P, animals, j1, [0] * len(animals), True)
        r2 = pooled_rr(P, animals, j2, [0] * len(animals), True)
        per[b] = float(np.log2(r1) - np.log2(r2))
    return {"stat_log2": float(obs), "p": float(p),
            "null_mean": float(v.mean()), "null_sd": float(v.std()),
            "n_behaviors": len(pairs), "per_behavior": per}


def precompute(S):
    """Per session and flag: circular correlations needed by A and B."""
    names = S[next(iter(S))]["names"]
    P = {}
    for animal, s in S.items():
        c = s["counts"]
        n = s["n"]
        entry = {"total": c.sum(), "n": n, "peri": {}, "in": {}, "strat": {}}
        for j, nm in enumerate(names):
            fl = s["flags"][:, j].astype(float)
            entry["peri"][j] = xcorr(c, fl)            # behaviour around calls
            entry["in"][j] = xcorr(fl, c)              # calls inside behaviour
            per_k = []
            for k in range(N_STRATA):
                m = (s["strat"] == k)
                f1 = (fl > 0) & m
                f0 = (fl == 0) & m
                per_k.append((xcorr(f1.astype(float), c),
                              xcorr(f0.astype(float), c),
                              f1.sum() * BIN_S, f0.sum() * BIN_S))
            entry["strat"][j] = per_k
            entry.setdefault("T1", {})[j] = fl.sum() * BIN_S
            entry.setdefault("T0", {})[j] = (fl == 0).sum() * BIN_S
        P[animal] = entry
    return P


# ------------------------------------------------------------ A enrichment
def pooled_rr(P, animals, j, shifts, stratified):
    """Mantel-Haenszel person-time rate ratio pooled over sessions (and, if
    `stratified`, over speed x distance strata within each session).

    Pooling at the estimator level - rather than averaging per-session log
    ratios - keeps sessions with zero calls in a state from blowing up, which
    a mean of logs cannot do.
    """
    num = den = 0.0
    for a, s in zip(animals, shifts):
        e = P[a]
        idx = (-s) % e["n"]
        if stratified:
            cells = [(c1[idx], c0[idx], t1, t0)
                     for (c1, c0, t1, t0) in e["strat"][j]]
        else:
            a_in = e["in"][j][idx]
            cells = [(a_in, e["total"] - a_in, e["T1"][j], e["T0"][j])]
        for (x1, x0, t1, t0) in cells:
            if t1 <= 0 or t0 <= 0:
                continue
            T = t1 + t0
            num += x1 * t0 / T
            den += x0 * t1 / T
    return (num / den) if den > 0 else np.nan


def enrichment(S, P, rng):
    names = S[next(iter(S))]["names"]
    rows = []
    zeros = [0] * len(S)
    for j, nm in enumerate(names):
        animals = [a for a in S
                   if S[a]["flags"][:, j].sum() >= 20
                   and (~S[a]["flags"][:, j]).sum() >= 20]
        if len(animals) < 6:
            continue
        z = [0] * len(animals)
        stat_c = np.log2(pooled_rr(P, animals, j, z, False))
        stat_a = np.log2(pooled_rr(P, animals, j, z, True))
        null_c = np.empty(N_PERM)
        null_a = np.empty(N_PERM)
        for p_ in range(N_PERM):
            sh = [int(rng.integers(1, P[a]["n"])) for a in animals]
            null_c[p_] = np.log2(pooled_rr(P, animals, j, sh, False))
            null_a[p_] = np.log2(pooled_rr(P, animals, j, sh, True))
        # how many sessions individually point the same way (robustness)
        pos = 0
        for a in animals:
            e = P[a]
            a_in = e["in"][j][0]
            t1, t0 = e["T1"][j], e["T0"][j]
            if t1 > 0 and t0 > 0 and e["total"] > 0:
                r1 = a_in / t1
                r0 = (e["total"] - a_in) / t0
                pos += int(r1 > r0)
        occ = np.mean([S[a]["flags"][:, j].mean() for a in animals]) * 100
        n_in = int(sum(S[a]["counts"][S[a]["flags"][:, j]].sum() for a in animals))

        def perm_p(null, stat):
            """Two-sided permutation p; NaN draws are dropped, not silently
            counted as non-exceedances."""
            v = null[np.isfinite(null)]
            if not len(v) or not np.isfinite(stat):
                return np.nan
            return (np.sum(np.abs(v - v.mean()) >= abs(stat - v.mean())) + 1) / (len(v) + 1)

        rows.append({
            "flag": nm, "actor": nm[:2], "behavior": nm[3:],
            "n_sessions": len(animals), "n_sessions_positive": pos,
            "occupancy_pct": occ, "calls_in_state": n_in,
            "n_null_valid": int(np.isfinite(null_a).sum()),
            "log2_RR_crude": stat_c, "p_crude": perm_p(null_c, stat_c),
            "log2_RR_adj": stat_a, "p_adj": perm_p(null_a, stat_a),
            "z_adj": (stat_a - np.nanmean(null_a)) / (np.nanstd(null_a) + 1e-9),
            "null_mean_adj": float(np.nanmean(null_a)),
        })
    out = pd.DataFrame(rows)
    out["q_crude"] = bh_fdr(out["p_crude"])
    out["q_adj"] = bh_fdr(out["p_adj"])
    return out.sort_values("log2_RR_adj", ascending=False)


# ------------------------------------------------------------ B peri-event
def peri_call(S, P, rng):
    half = int(PERI_HALF_S / BIN_S)
    lags = np.arange(-half, half + 1)
    names = S[next(iter(S))]["names"]
    rows = []
    for j, nm in enumerate(names):
        animals = [a for a in S if S[a]["counts"].sum() >= 5]
        w = np.array([P[a]["total"] for a in animals], float)
        n_lag = len(lags)
        # peri[j] already holds the SUM over calls of the flag value, so the
        # pooled mean is sum-of-sums / total number of calls
        obs = np.zeros(n_lag)
        for a in animals:
            n = P[a]["n"]
            obs += P[a]["peri"][j][lags % n]
        obs /= w.sum()
        null = np.empty((N_PERM, n_lag))
        for p_ in range(N_PERM):
            acc = np.zeros(n_lag)
            for a in animals:
                n = P[a]["n"]
                sh = int(rng.integers(1, n))
                acc += P[a]["peri"][j][(lags + sh) % n]
            null[p_] = acc / w.sum()
        lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
        for k in range(n_lag):
            rows.append({"flag": nm, "lag_s": lags[k] * BIN_S,
                         "p_behavior": obs[k], "null_lo": lo[k],
                         "null_hi": hi[k], "null_mean": null[:, k].mean(),
                         "n_calls": int(w.sum())})
    return pd.DataFrame(rows)


def merge_bouts(flag, max_gap_bins=3, min_len_bins=1):
    v = flag.astype(int)
    d = np.diff(np.r_[0, v, 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    if not len(starts):
        return np.array([], int)
    keep_s, keep_e = [starts[0]], []
    for i in range(1, len(starts)):
        if starts[i] - ends[i - 1] <= max_gap_bins:
            continue
        keep_e.append(ends[i - 1])
        keep_s.append(starts[i])
    keep_e.append(ends[-1])
    s, e = np.array(keep_s), np.array(keep_e)
    return s[(e - s) >= min_len_bins]


def peri_onset(S, rng):
    half = int(PERI_HALF_S / BIN_S)
    lags = np.arange(-half, half + 1)
    names = S[next(iter(S))]["names"]
    rows = []
    for j, nm in enumerate(names):
        corr, weights = [], []
        for animal, s in S.items():
            on = merge_bouts(s["flags"][:, j])
            if len(on) < 3:
                continue
            mask = np.zeros(s["n"])
            mask[on] = 1.0
            corr.append(xcorr(mask, s["counts"]))
            weights.append(len(on))
        if sum(weights) < 20:
            continue
        w = np.array(weights, float)
        n_lag = len(lags)
        obs = np.zeros(n_lag)
        for c, wa in zip(corr, w):
            obs += c[lags % len(c)]
        obs = obs / w.sum() / BIN_S * 60.0
        null = np.empty((N_PERM, n_lag))
        for p_ in range(N_PERM):
            acc = np.zeros(n_lag)
            for c in corr:
                sh = int(rng.integers(1, len(c)))
                acc += c[(lags + sh) % len(c)]
            null[p_] = acc / w.sum() / BIN_S * 60.0
        lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
        for k in range(n_lag):
            rows.append({"flag": nm, "lag_s": lags[k] * BIN_S,
                         "call_rate": obs[k], "null_lo": lo[k],
                         "null_hi": hi[k], "n_onsets": int(w.sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------- C consequence
WINDOWS = {"early_0.2-1s": (0.2, 1.0), "mid_0.2-2s": (0.2, 2.0),
           "late_2-5s": (2.0, 5.0)}


def consequence(df, rng, conf=0.65):
    pre0, pre1 = int(1.0 / BIN_S), int(0.2 / BIN_S)
    guard = int(1.0 / BIN_S)
    max_post = int(max(w[1] for w in WINDOWS.values()) / BIN_S)
    metrics = KIN_COLS + ["m1_chasing", "m1_following", "m1_approach",
                          "m2_escape", "m2_withdrawal_from_partner",
                          "m1_nose2anogenital", "m1_nose2nose",
                          "m2_approach", "m2_following"]
    rows = []
    for animal, g in df[df["t"] >= PARTNER_T0].groupby("animal_id"):
        g = g.sort_values("bin").reset_index(drop=True)
        n = len(g)
        M = {m: g[m].to_numpy(float) for m in metrics}
        sp, di = g["speed_m1"].to_numpy(), g["dist"].to_numpy()
        top = g["behavior_top"].fillna("none").astype(str).to_numpy()
        counts = g["n_calls"].to_numpy()
        pres = g["p_res"].to_numpy()

        def qb(v, q):
            v = np.where(np.isfinite(v), v, np.nanmedian(v))
            return np.digitize(v, np.nanquantile(v, q))
        key = np.array([f"{a}|{b}{c}" for a, b, c in
                        zip(top, qb(sp, [.2, .4, .6, .8]), qb(di, [1/3, 2/3]))])
        has_call = counts > 0
        near = np.convolve(has_call.astype(int), np.ones(2 * guard + 1),
                           mode="same") > 0
        pool = {}
        for i in np.where(~near)[0]:
            if pre0 <= i < n - max_post:
                pool.setdefault(key[i], []).append(i)
        call_idx = np.where(has_call)[0]
        call_idx = call_idx[(call_idx >= pre0) & (call_idx < n - max_post)]
        if len(call_idx) < 5:
            continue

        def delta(i, m, w0, w1):
            return (np.nanmean(M[m][i + w0:i + w1])
                    - np.nanmean(M[m][i - pre0:i - pre1]))

        for i in call_idx:
            cands = pool.get(key[i], [])
            if len(cands) < 2:
                continue
            ctrl = rng.choice(cands, size=min(3, len(cands)), replace=False)
            role = ("res" if pres[i] >= conf else
                    "part" if pres[i] <= 1 - conf else "unsure")
            for win, (a0, a1) in WINDOWS.items():
                w0, w1 = int(a0 / BIN_S), int(a1 / BIN_S)
                for m in metrics:
                    dc = delta(i, m, w0, w1)
                    dk = np.nanmean([delta(int(j), m, w0, w1) for j in ctrl])
                    if np.isfinite(dc) and np.isfinite(dk):
                        rows.append({"animal_id": animal,
                                     "genotype": A.GENOTYPE_MAP[animal],
                                     "metric": m, "role": role, "window": win,
                                     "d_call": dc, "d_ctrl": dk,
                                     "diff": dc - dk})
    return pd.DataFrame(rows)


def consequence_stats(cons, rng, n_boot=2000):
    """Session-level Wilcoxon (conservative) plus a cluster bootstrap over
    sessions, which keeps call-level precision without pretending calls
    within a session are independent."""
    rows = []
    for (m, role, win), g in cons.groupby(["metric", "role", "window"]):
        per = g.groupby("animal_id")["diff"].mean()
        if len(per) < 6:
            continue
        p = stats.wilcoxon(per)[1] if per.abs().sum() > 0 else np.nan
        animals = g["animal_id"].unique()
        by = {a: g.loc[g["animal_id"] == a, "diff"].to_numpy() for a in animals}
        boot = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.choice(animals, len(animals), replace=True)
            boot[b] = np.concatenate([by[a] for a in pick]).mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pooled = g["diff"].mean()
        p_boot = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
        rows.append({"metric": m, "role": role, "window": win,
                     "n_sessions": len(per), "n_calls": int(len(g)),
                     "mean_diff_sessions": float(per.mean()),
                     "median_diff_sessions": float(per.median()),
                     "pooled_diff": float(pooled),
                     "boot_lo": float(lo), "boot_hi": float(hi),
                     "p_wilcoxon": p, "p_boot": float(max(p_boot, 1 / n_boot))})
    out = pd.DataFrame(rows)
    for (role, win), g in out.groupby(["role", "window"]):
        out.loc[g.index, "q_boot"] = bh_fdr(g["p_boot"].to_numpy())
    return out.sort_values(["role", "window", "p_boot"])


# ------------------------------------------------------------ D prediction
def prediction(S):
    names = S[next(iter(S))]["names"]
    animals = list(S)
    Xb, Xk, y, grp = [], [], [], []
    for animal, s in S.items():
        Xb.append(s["flags"].astype(float))
        k = s["g"][KIN_COLS].to_numpy(float)
        k = np.where(np.isfinite(k), k, np.nanmedian(k, axis=0))
        k = (k - k.mean(0)) / (k.std(0) + 1e-9)
        Xk.append(k)
        y.append((s["counts"] > 0).astype(int))
        grp.append(np.repeat(animal, s["n"]))
    Xb, Xk = np.vstack(Xb), np.vstack(Xk)
    y, grp = np.concatenate(y), np.concatenate(grp)
    sets = {"kinematics": Xk, "behaviour": Xb, "both": np.hstack([Xk, Xb])}
    res = {}
    for tag, X in sets.items():
        aucs = []
        for a in animals:
            te = grp == a
            if y[te].sum() < 5 or y[~te].sum() < 20:
                continue
            clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                                     C=0.5).fit(X[~te], y[~te])
            aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
        res[tag] = {"loso_auc_mean": float(np.mean(aucs)),
                    "loso_auc_sd": float(np.std(aucs)),
                    "n_sessions": len(aucs),
                    "per_session": [float(x) for x in aucs]}
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             C=0.5).fit(sets["both"], y)
    res["coefficients_both"] = dict(zip(KIN_COLS + names,
                                        clf.coef_[0].round(3).tolist()))
    res["n_bins"] = int(len(y))
    res["n_call_bins"] = int(y.sum())
    return res


# ------------------------------------------------------------- E call types
def call_types(df, S, rng):
    part = df[df["t"] >= PARTNER_T0]
    calls = part[part["n_calls"] > 0].copy()
    calls["dur_tercile"] = pd.qcut(calls["call_dur_ms"], 3,
                                   labels=["short", "mid", "long"])
    # rebuild the composite flags that session_arrays adds to its matrix
    for tag in ("m1", "m2"):
        calls[f"{tag}_ACTIVEsocial"] = np.maximum.reduce(
            [calls[f"{tag}_{b}"].to_numpy() for b in ACTIVE_SET])
    calls["m1_CONTACTmutual"] = np.maximum.reduce(
        [calls[f"m1_{b}"].to_numpy()
         for b in ("nose2nose", "sidebyside", "sidereside")])
    names = S[next(iter(S))]["names"]
    rows = []
    for scheme, col in (("syllable", "call_class"), ("duration", "dur_tercile")):
        vc = calls[col].value_counts()
        for typ in vc[vc >= 40].index:
            sel = calls[calls[col] == typ]
            for nm in names:
                p_t = float((sel[nm] > 0).mean())
                p_a = float((calls[nm] > 0).mean())
                rows.append({"scheme": scheme, "type": str(typ),
                             "n": int(len(sel)), "flag": nm, "p_in_state": p_t,
                             "p_all_calls": p_a,
                             "log2_ratio": np.log2((p_t + 5e-3) / (p_a + 5e-3))})
    prof = pd.DataFrame(rows)
    het = {}
    for scheme, col in (("syllable", "call_class"), ("duration", "dur_tercile")):
        sub = calls[calls[col].notna()].copy()
        vc = sub[col].value_counts()
        sub = sub[sub[col].isin(vc[vc >= 40].index)]
        if sub[col].nunique() < 2:
            continue
        obs = _het(sub, names, col)
        null = [_het(sub.assign(**{col: rng.permutation(sub[col].to_numpy())}),
                     names, col) for _ in range(400)]
        het[scheme] = {"stat": float(obs),
                       "p": float((np.sum(np.array(null) >= obs) + 1) / 401),
                       "null_mean": float(np.mean(null)),
                       "n_types": int(sub[col].nunique()),
                       "n_calls": int(len(sub))}
    return prof, het


def _het(sub, names, col):
    M = np.stack([(sub[sub[col] == t][names] > 0).mean().to_numpy()
                  for t in sub[col].unique()])
    return float(np.nanmean(np.nanstd(M, axis=0)))


def _profile(S, P, animals, j):
    use = [a for a in animals if S[a]["flags"][:, j].sum() >= 20]
    if len(use) < 3:
        return np.nan
    r = pooled_rr(P, use, j, [0] * len(use), False)
    return np.log2(r) if (np.isfinite(r) and r > 0) else np.nan


# ------------------------------------------------------------------- main
def main():
    out = A.ensure_out()
    rng = np.random.default_rng(SEED)
    df = load_bins()
    S = session_arrays(df)
    P = precompute(S)
    n_calls = int(sum(s["counts"].sum() for s in S.values()))
    print(f"{len(S)} sessions, {n_calls} partner-phase calls")

    print("A) enrichment ...", flush=True)
    enr = enrichment(S, P, rng)
    enr.to_csv(out / "beh_enrichment.csv", index=False)
    print(enr[["flag", "occupancy_pct", "calls_in_state", "log2_RR_crude",
               "log2_RR_adj", "q_adj"]].head(14).to_string(index=False))

    print("A2) actor asymmetry ...", flush=True)
    asym = actor_asymmetry(S, P, rng)
    print(f"   calls track the actor: log2 = {asym['stat_log2']:+.2f}, "
          f"p = {asym['p']:.4f}")

    print("B) peri-event ...", flush=True)
    peri_call(S, P, rng).to_csv(out / "beh_peri_call.csv", index=False)
    peri_onset(S, rng).to_csv(out / "beh_peri_onset.csv", index=False)

    print("C) consequences ...", flush=True)
    cons = consequence(df, rng)
    cons.to_csv(out / "beh_consequence.csv", index=False)
    cst = consequence_stats(cons, rng)
    cst.to_csv(out / "beh_consequence_stats.csv", index=False)
    print(cst[cst["role"] == "res"].head(16)[
        ["metric", "window", "pooled_diff", "boot_lo", "boot_hi",
         "p_boot", "q_boot"]].to_string(index=False))

    print("D) prediction ...", flush=True)
    pred = prediction(S)
    with open(out / "beh_prediction.json", "w") as fh:
        json.dump(pred, fh, indent=2)
    print({k: pred[k]["loso_auc_mean"] for k in
           ("kinematics", "behaviour", "both")})

    print("E) call types ...", flush=True)
    prof, het = call_types(df, S, rng)
    prof.to_csv(out / "beh_calltype.csv", index=False)

    animals = list(S)
    names = S[animals[0]]["names"]
    rhos = []
    for _ in range(200):
        perm = rng.permutation(animals)
        h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
        e1 = np.array([_profile(S, P, h1, j) for j in range(len(names))])
        e2 = np.array([_profile(S, P, h2, j) for j in range(len(names))])
        ok = np.isfinite(e1) & np.isfinite(e2)
        if ok.sum() > 5:
            rhos.append(stats.spearmanr(e1[ok], e2[ok])[0])
    val = {"splithalf_rho_mean": float(np.nanmean(rhos)),
           "splithalf_rho_lo": float(np.nanpercentile(rhos, 2.5)),
           "splithalf_rho_hi": float(np.nanpercentile(rhos, 97.5)),
           "n_perm": N_PERM, "calltype_heterogeneity": het,
           "actor_asymmetry": asym,
           "n_sessions": len(S), "n_calls_partner": n_calls,
           "note_behavior_labels": ("rule-based on the same DLC keypoints as "
                                    "the GEO attribution: consistent-with, not "
                                    "independent-of")}
    with open(out / "beh_validation.json", "w") as fh:
        json.dump(val, fh, indent=2)
    print(json.dumps(val, indent=2))


if __name__ == "__main__":
    main()
