"""One pre-specified directionality test instead of 25 underpowered ones.

v4 established that calls are enriched while the resident is actively
investigating.  That makes the resident's investigation suite the a-priori
target, so the composite is pre-specified rather than picked after looking.
The partner's flight suite is the natural counterpart for "the resident calls
because the partner ran".
"""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import attr_common as A
import attr5_core as C
from attr5_10_directionality import profile_block

SUITES = {
    "resident_investigation": ("m1", ["nose2anogenital", "nose2body",
                                      "following", "chasing", "approach",
                                      "oriented_toward"]),
    "partner_flight": ("m2", ["escape", "withdrawal_from_partner",
                              "withdrawal_after_contact"]),
    "partner_investigation": ("m2", ["nose2anogenital", "nose2body",
                                     "following", "chasing", "approach",
                                     "oriented_toward"]),
    "social_contact": ("m1", ["nose2nose", "sidebyside", "sidereside"]),
}


def main():
    out = A.ensure_out()
    bins = C.load_bins()
    calls = pd.read_csv(out / "v5_calls.csv", low_memory=False)
    sess = C.session_arrays(bins, calls)
    wt = [k for k, v in sess.items() if v["genotype"] == "WT"]

    for d in sess.values():
        for name, (who, flags) in SUITES.items():
            v = np.zeros(d["n"])
            for f in flags:
                v = np.maximum(v, d[f"{who}_{f}"])
            d[f"suite_{name}"] = v

    res, profiles = {}, {}
    print("=== pre-specified composite directionality (WT) ===")
    print("  Lambda < 0: behaviour leads the call     Lambda > 0: call leads\n")
    for name in SUITES:
        r = profile_block(sess, wt, "x_res", f"suite_{name}", lmax=50,
                          n_draw=8000)
        if r is None:
            continue
        res[name] = {k: (float(v) if np.isscalar(v) else None)
                     for k, v in r.items()
                     if k in ("lambda", "z", "p", "occupancy", "n_calls",
                              "n_sessions", "latency_s", "lam_null_mean",
                              "lam_null_sd")}
        res[name]["p_before"] = float(np.mean(r["before"][:20]))
        res[name]["p_after"] = float(np.mean(r["after"][:20]))
        res[name]["enrichment"] = float(np.mean(r["even"][:20]) / r["occupancy"])
        profiles[name] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in r.items()}
        print(f"  {name:>22}: occupancy {100*r['occupancy']:4.1f}%   "
              f"enrichment {res[name]['enrichment']:.2f}x   "
              f"Lambda {r['lambda']:+.2f}  z {r['z']:+.2f}  p {r['p']:.4f}")
    ps = np.array([res[k]["p"] for k in res])
    qs = C.bh(ps)
    for k, q in zip(res, qs):
        res[k]["q"] = float(q)
    print(f"\n  after FDR over {len(res)} pre-specified suites: "
          f"{int((qs < 0.05).sum())} significant")

    json.dump({"results": res, "profiles": profiles},
              open(out / "v5_composite_direction.json", "w"))
    print("wrote", out / "v5_composite_direction.json")


if __name__ == "__main__":
    main()
