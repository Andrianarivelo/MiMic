"""Apply the attribution model to the real LgDel sessions.

Builds the voice space (handcrafted + from-scratch AE latents), fits the
intensity-distance model on alone-phase calls, then runs the hierarchical
two-source EM with partner-voice tying and bout-HMM smoothing.

Outputs:
  attribution/call_attribution.csv   per-call posteriors (partner-phase calls;
                                     alone-phase calls carry p=1 by design)
  attribution/attribution_model.npz  fitted parameters for figures
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A
import attr_core as CORE


def load_joined():
    out = A.ensure_out()
    feats = pd.read_csv(out / "call_features.csv")
    emb = np.load(out / "call_embeddings.npz", allow_pickle=False)
    lat = pd.DataFrame(emb["latent"])
    lat.columns = [f"z{i}" for i in range(lat.shape[1])]
    lat["call_id"] = emb["call_id"]
    df = feats.merge(lat, on="call_id", how="inner")
    df = A.apply_genotype(df)
    return df, [c for c in df.columns if c.startswith("z") and c[1:].isdigit()]


def main():
    out = A.ensure_out()
    df, zcols = load_joined()
    df["animal_id"] = df["animal_id"].astype(str)
    print(f"{len(df)} calls with features+latents")

    X, transform, pca = CORE.build_voice_space(df, df[zcols].to_numpy())

    alone = df["phase"] == "alone"
    imodel = CORE.fit_intensity_model(
        level_db=df.loc[alone, "rf_amp_peak_db"].to_numpy(),
        pos_xy=df.loc[alone, ["m1_x", "m1_y"]].to_numpy(),
        animal_idx=df.loc[alone, "animal_id"].to_numpy(),
    )
    print(f"intensity model: R2={imodel.r2_insample:.3f}, "
          f"LOAO R2={imodel.r2_loao:.3f}, enabled={imodel.enabled}, "
          f"mic=({imodel.mic[0]:.0f},{imodel.mic[1]:.0f})px, b={imodel.b:.2f}")

    sessions = []
    for animal, g in df.groupby("animal_id"):
        part = g[g["phase"] == "partner"]
        anch = g[g["phase"] == "alone"]
        idxX = part.index.to_numpy()
        stim = A.STIM_MAP.get(animal) or f"solo_{animal}"
        sessions.append(CORE.Session(
            key=animal, stim_key=stim,
            X=X[idxX], times=part["start_s"].to_numpy(),
            level_db=part["rf_amp_peak_db"].to_numpy(),
            pos_R=part[["m1_x", "m1_y"]].to_numpy(),
            pos_P=part[["m2_x", "m2_y"]].to_numpy(),
            anchor_X=X[anch.index.to_numpy()],
        ))

    res = CORE.fit_attribution(sessions, imodel, seed=A.RNG_SEED)

    rows = []
    for s in sessions:
        g = df[(df["animal_id"] == s.key) & (df["phase"] == "partner")]
        r = res[s.key]
        for j, (_, c) in enumerate(g.iterrows()):
            rows.append({
                "call_id": c["call_id"], "animal_id": s.key,
                "genotype": c["genotype"], "phase": "partner",
                "start_s": c["start_s"], "stim_key": s.stim_key,
                "p_res_voice": r["p_res_voice"][j],
                "p_res_full": r["p_res_full"][j],
                "p_res_hmm": r["p_res_hmm"][j],
                "session_pi": r["pi"],
            })
    for _, c in df[alone].iterrows():
        rows.append({
            "call_id": c["call_id"], "animal_id": c["animal_id"],
            "genotype": c["genotype"], "phase": "alone",
            "start_s": c["start_s"],
            "stim_key": A.STIM_MAP.get(c["animal_id"]) or f"solo_{c['animal_id']}",
            "p_res_voice": 1.0, "p_res_full": 1.0, "p_res_hmm": 1.0,
            "session_pi": np.nan,
        })
    att = pd.DataFrame(rows).sort_values(["animal_id", "start_s"])
    att.to_csv(out / "call_attribution.csv", index=False)

    np.savez_compressed(
        out / "attribution_model.npz",
        mic=imodel.mic, b=imodel.b, a0=imodel.a0, sigma=imodel.sigma,
        r2_insample=imodel.r2_insample, r2_loao=imodel.r2_loao,
        intensity_enabled=imodel.enabled,
        pca_explained=pca.explained_variance_ratio_,
        muR=np.stack([res[s.key]["muR"] for s in sessions]),
        muP=np.stack([res[s.key]["muP"] for s in sessions]),
        session_keys=np.array([s.key for s in sessions]),
        pi=np.array([res[s.key]["pi"] for s in sessions]),
        varR=res["_globals"]["varR"], varP=res["_globals"]["varP"],
    )

    part = att[att["phase"] == "partner"]
    for gt in ("WT", "HET"):
        sub = part[part["genotype"] == gt]
        print(f"{gt}: {len(sub)} partner-phase calls, "
              f"resident fraction (HMM)={sub['p_res_hmm'].mean():.2f}")
    print(f"wrote {out / 'call_attribution.csv'}")


if __name__ == "__main__":
    main()
