"""Build 100 ms behaviour x kinematics x call tables for all 24 sessions.

Source: `*_live_detections.csv` (PyKaboo live rule engine, `behavior_backend
= "rules"`). IMPORTANT: these labels are geometric rules over the same DLC
keypoints that the v2 GEO attribution uses, so they are a different
formalisation of the same tracking stream - NOT an independent modality. The
CALLS, however, are an independent acoustic measurement, so call-behaviour
coupling is a genuine cross-modal result.

Flag semantics (verified empirically):
  mutual states  (nose2nose, sidebyside, sidereside, fighting) - identical on
                 both mice
  directional    (nose2anogenital, nose2body, oriented_toward, following,
                 chasing, approach, withdrawal_*, escape) - marked on the
                 ACTOR, so m1_chasing means the resident is the chaser
  solitary       (rearing, passive)

Outputs: attribution/beh_bins.csv.gz  (one row per 100 ms bin per session)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A
import attr2_core as C2
from attr2_45_geometry import session_geometry

BEH = ["nose2nose", "sidebyside", "sidereside", "nose2anogenital", "nose2body",
       "oriented_toward", "following", "chasing", "approach",
       "withdrawal_from_partner", "escape", "withdrawal_after_contact",
       "fighting", "rearing", "passive"]
MUTUAL = {"nose2nose", "sidebyside", "sidereside", "fighting"}
SOLITARY = {"rearing", "passive"}
DIRECTIONAL = [b for b in BEH if b not in MUTUAL and b not in SOLITARY]

BIN_S = 0.1
FPS_EXACT = 30.000063
LEAK_CUTOFF_S = 295.0     # dyadic flags before this are detector artefacts


def session_bins(animal: str, calls: pd.DataFrame, kin: pd.DataFrame,
                 geo_post: pd.DataFrame) -> pd.DataFrame:
    cols = ["frame_id", "mouse_id", "behavior_top"] + [f"behavior_{b}" for b in BEH]
    d = pd.read_csv(A.live_detections_path(animal), usecols=cols, low_memory=False)
    d["t"] = d["frame_id"] / FPS_EXACT
    d["bin"] = np.floor(d["t"] / BIN_S).astype(int)

    frames = {}
    for mid, tag in ((1, "m1"), (2, "m2")):
        sub = d[d["mouse_id"] == mid]
        if not len(sub):
            continue
        g = sub.groupby("bin")[[f"behavior_{b}" for b in BEH]].max()
        g.columns = [f"{tag}_{b}" for b in BEH]
        frames[tag] = g
    top = (d[d["mouse_id"] == 1].groupby("bin")["behavior_top"]
           .agg(lambda s: s.mode().iat[0] if len(s.mode()) else "none"))

    n_bins = int(A.SESSION_END_S / BIN_S)
    out = pd.DataFrame({"bin": np.arange(n_bins)})
    out["t"] = out["bin"] * BIN_S + BIN_S / 2
    for tag in ("m1", "m2"):
        if tag in frames:
            out = out.merge(frames[tag], on="bin", how="left")
        else:
            for b in BEH:
                out[f"{tag}_{b}"] = 0.0
    out["behavior_top"] = out["bin"].map(top).fillna("none")
    beh_cols = [c for c in out.columns if c.startswith(("m1_", "m2_"))]
    out[beh_cols] = out[beh_cols].fillna(0.0)

    # kill dyadic-flag leakage in the alone phase (partner not yet present)
    pre = out["t"] < LEAK_CUTOFF_S
    dy = [c for c in beh_cols
          if c.split("_", 1)[1] not in SOLITARY]
    out.loc[pre, dy] = 0.0
    out.loc[pre, "behavior_top"] = out.loc[pre, "behavior_top"].where(
        out.loc[pre, "behavior_top"].isin(["none", "rearing", "passive"]), "none")

    # kinematics + social geometry on the same grid
    t = out["t"].to_numpy()
    geo = session_geometry(kin, t)
    out["speed_m1"] = geo["s1"]
    out["speed_m2"] = geo["s2"]
    out["dist"] = geo["dist"]
    out["app_m1"] = geo["app1"]
    out["app_m2"] = geo["app2"]

    # calls
    g = calls[calls["animal_id"] == animal]
    ct = ((g["start_s"] + g["end_s"]) / 2).to_numpy()
    cbin = np.floor(ct / BIN_S).astype(int)
    cnt = pd.Series(cbin).value_counts()
    out["n_calls"] = out["bin"].map(cnt).fillna(0).astype(int)
    # call class + duration of the (first) call in the bin
    cls = pd.Series(g["class_top1"].to_numpy(), index=cbin)
    dur = pd.Series((g["end_s"] - g["start_s"]).to_numpy() * 1e3, index=cbin)
    out["call_class"] = out["bin"].map(cls[~cls.index.duplicated()])
    out["call_dur_ms"] = out["bin"].map(dur[~dur.index.duplicated()])
    # v2 GEO posterior (probability the resident emitted it)
    gp = geo_post[geo_post["animal_id"] == animal]
    if len(gp):
        pb = pd.Series(gp["p_res_hmm"].to_numpy(),
                       index=np.floor(gp["t"].to_numpy() / BIN_S).astype(int))
        out["p_res"] = out["bin"].map(pb[~pb.index.duplicated()])
    else:
        out["p_res"] = np.nan

    out["animal_id"] = animal
    out["genotype"] = A.GENOTYPE_MAP[animal]
    return out


def main():
    out_dir = A.ensure_out()
    calls = A.load_calls()
    kin, _, _ = C2.load_kinematics()
    geo_post = pd.read_csv(out_dir / "v2_geo_attribution.csv")
    geo_post["animal_id"] = geo_post["animal_id"].astype(str)

    parts = []
    for i, animal in enumerate(A.GENOTYPE_MAP, 1):
        parts.append(session_bins(animal, calls, kin[animal], geo_post))
        print(f"[{i}/24] {animal}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out_dir / "beh_bins.csv.gz", index=False, compression="gzip")
    print(f"wrote {len(df)} bins -> beh_bins.csv.gz")

    part = df[df["t"] >= 305]
    print("\npartner-phase occupancy (% of bins, resident actor):")
    for b in BEH:
        print(f"  {b:26s} {100*part[f'm1_{b}'].mean():5.2f}%   "
              f"calls in state: {int(part.loc[part[f'm1_{b}']>0, 'n_calls'].sum()):5d}")


if __name__ == "__main__":
    main()
