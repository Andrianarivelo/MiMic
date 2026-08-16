"""Checkpoint-free call typing.

The upstream syllable labels come from a pretrained VocalPy classifier, which
this project is not allowed to depend on.  Here the repertoire is rebuilt from
our own spectrogram-ridge measurements: standardise, PCA, then a Gaussian
mixture whose component count is chosen by BIC.  Nothing is pretrained and
nothing is fit to genotype.
"""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import attr_common as A
import attr5_core as C

FEATS = [
    "rf_dur_ms", "rf_f_mean_khz", "rf_f_std_khz", "rf_slope_khz_ms",
    "rf_abs_slope_khz_ms", "rf_curve_khz_ms2", "rf_sinuosity", "rf_jumps",
    "rf_frac_rising", "rf_entropy_bits", "rf_spread_khz", "rf_tonality",
    "rf_amp_cv", "rf_amp_peak_pos",
]


def main():
    out = A.ensure_out()
    calls = C.load_calls_ctx()
    X = calls[FEATS].to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    print(f"{len(calls)} partner-window calls, {ok.sum()} with complete ridge features")

    Z = StandardScaler().fit_transform(X[ok])
    # log-transform the two heavy-tailed positive features before scaling
    pca = PCA(n_components=6, random_state=A.RNG_SEED)
    P = pca.fit_transform(Z)
    print("PCA explained variance:", np.round(pca.explained_variance_ratio_, 3))

    rows = []
    best, best_bic = None, np.inf
    for k in range(2, 9):
        gm = GaussianMixture(k, covariance_type="full", n_init=8,
                             random_state=A.RNG_SEED).fit(P)
        bic = gm.bic(P)
        rows.append({"k": k, "bic": bic, "aic": gm.aic(P)})
        print(f"  K={k}  BIC={bic:,.0f}")
        if bic < best_bic:
            best, best_bic = gm, bic
    K = best.n_components
    lab = best.predict(P)
    print(f"BIC selects K = {K}")

    calls["ctype"] = -1
    calls.loc[ok, "ctype"] = lab
    calls["ctype_p"] = np.nan
    calls.loc[ok, "ctype_p"] = best.predict_proba(P).max(axis=1)

    # describe each cluster by the standardised feature means, and name it
    prof = []
    for k in range(K):
        m = calls["ctype"] == k
        row = {"ctype": k, "n": int(m.sum())}
        for f in FEATS:
            row[f] = float(calls.loc[m, f].median())
        row["z_dur"] = float(np.mean(Z[lab == k, FEATS.index("rf_dur_ms")]))
        row["z_slope"] = float(np.mean(Z[lab == k, FEATS.index("rf_slope_khz_ms")]))
        row["z_tonal"] = float(np.mean(Z[lab == k, FEATS.index("rf_tonality")]))
        row["z_fmean"] = float(np.mean(Z[lab == k, FEATS.index("rf_f_mean_khz")]))
        prof.append(row)
    prof = pd.DataFrame(prof).sort_values("rf_dur_ms").reset_index(drop=True)

    # human-readable names: duration band x frequency-modulation sign
    def dur_tag(ms):
        return "ultrashort" if ms < 9 else ("short" if ms < 20 else
                                            ("long" if ms < 38 else "very long"))

    names = {}
    for _, r in prof.iterrows():
        k = int(r["ctype"])
        sl = r["rf_slope_khz_ms"]
        tag_fm = "up-FM" if sl > 0.2 else ("down-FM" if sl < -0.2 else "flat")
        names[k] = f"{dur_tag(r['rf_dur_ms'])} {tag_fm}"
        print(f"  type {k}: {names[k]:>18}  n={int(r['n']):4d}  "
              f"dur={r['rf_dur_ms']:5.1f} ms  f={r['rf_f_mean_khz']:5.1f} kHz  "
              f"slope={sl:+.3f} kHz/ms  tonality={r['rf_tonality']:.2f}")
    prof["name"] = prof["ctype"].map(names)
    calls["ctype_name"] = calls["ctype"].map(names)

    calls.to_csv(out / "v5_calls.csv", index=False)
    prof.to_csv(out / "v5_calltype_profile.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "v5_calltype_bic.csv", index=False)
    json.dump({"K": int(K), "names": {str(k): v for k, v in names.items()},
               "explained_variance": pca.explained_variance_ratio_.tolist(),
               "n_calls": int(len(calls)), "n_typed": int(ok.sum())},
              open(out / "v5_calltype_model.json", "w"), indent=2)

    print("\ncomposition by genotype:")
    print(pd.crosstab(calls["ctype_name"], calls["genotype"]).to_string())
    print("wrote", out / "v5_calls.csv")


if __name__ == "__main__":
    main()
