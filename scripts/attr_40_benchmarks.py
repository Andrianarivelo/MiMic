"""Ground-truth benchmarks and validations for single-mic call attribution.

The only calls with certain speaker identity are alone-phase calls (0-300 s,
resident alone in the arena). They power four benchmarks:

A. IDENTITY SIGNAL - is there an individual voice fingerprint at all?
   Pairwise same-animal vs different-animal distance AUC in voice space +
   leave-one-out nearest-centroid speaker ID with a permutation null.
B. FEATURE-LEVEL PSEUDO-DYADS - for animal pairs (A resident, B partner):
   anchor = first half of A's alone calls; test "session" = second half of A
   + all of B's alone calls, with each call's true position taken from its
   own session's tracking. Run the EXACT production model; score AUC,
   accuracy, and mixing-weight recovery against ground truth.
C. AUDIO-LEVEL MIXTURES - same pairs, but features are re-extracted after
   summing the two 384 kHz waveforms (real interference + noise conditions),
   using the same from-scratch AE for latents. End-to-end robustness.
D. NEGATIVE CONTROL - real sessions refitted with half the anchors; held-out
   alone calls (guaranteed resident) must attribute to the resident.

Also: timestamp collision rates and pooled calibration (error rates at 0.5
used by attr_50 to correct session-level fractions).

Outputs under attribution/: benchmark_pairs.csv, benchmark_calls.csv,
benchmark_negative_control.csv, benchmark_summary.json, benchmark_curves.npz
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A
import attr_core as CORE
import attr_10_extract_features as EX

MIN_RES_CALLS = 6      # resident needs anchor half + test half
MIN_PART_CALLS = 3
N_AUDIO_PAIRS = 12     # audio-level mixtures are slower; use the best pairs
SEED = A.RNG_SEED


# ------------------------------------------------------------------ helpers
def load_all():
    out = A.ensure_out()
    feats = pd.read_csv(out / "call_features.csv")
    emb = np.load(out / "call_embeddings.npz", allow_pickle=False)
    lat = pd.DataFrame(emb["latent"])
    zcols = [f"z{i}" for i in range(lat.shape[1])]
    lat.columns = zcols
    lat["call_id"] = emb["call_id"]
    df = feats.merge(lat, on="call_id", how="inner")
    df["animal_id"] = df["animal_id"].astype(str)
    df = A.apply_genotype(df)
    return df, zcols


def fit_spaces(df, zcols):
    X, transform, _ = CORE.build_voice_space(df, df[zcols].to_numpy())
    alone = df["phase"] == "alone"
    imodel = CORE.fit_intensity_model(
        level_db=df.loc[alone, "rf_amp_peak_db"].to_numpy(),
        pos_xy=df.loc[alone, ["m1_x", "m1_y"]].to_numpy(),
        animal_idx=df.loc[alone, "animal_id"].to_numpy(),
    )
    return X, transform, imodel


def score_with_model(res_entry, X, times, level, posR, posP, imodel, globals_):
    """Score new calls with a fitted session model (used by negative control)."""
    varR, varP = globals_["varR"], globals_["varP"]
    llR = CORE._gauss_logpdf_diag(X, res_entry["muR"], varR)
    llP = CORE._gauss_logpdf_diag(X, res_entry["muP"], varP)
    sess = CORE.Session("tmp", "tmp", X, times, level, posR, posP,
                        np.zeros((0, X.shape[1])))
    iR, iP = CORE._intensity_loglik(sess, imodel)
    pi = res_entry["pi"]
    a = np.log(pi) + llR + iR
    b = np.log(1 - pi) + llP + iP
    return 1.0 / (1.0 + np.exp(np.clip(b - a, -500, 500)))


# ------------------------------------------------------- A. identity signal
def identity_signal(df, X):
    alone = df[df["phase"] == "alone"]
    idx = alone.index.to_numpy()
    Xa = X[idx]
    ids = alone["animal_id"].to_numpy()
    rng = np.random.default_rng(SEED)

    d2 = np.sum((Xa[:, None, :] - Xa[None, :, :]) ** 2, axis=2)
    iu = np.triu_indices(len(Xa), k=1)
    same = (ids[iu[0]] == ids[iu[1]]).astype(int)
    dist = d2[iu]
    auc_same = roc_auc_score(same, -dist) if same.sum() else np.nan

    keep_animals = [a for a, n in alone.groupby("animal_id").size().items()
                    if n >= 4]
    mask = np.isin(ids, keep_animals)
    Xs, ys = Xa[mask], ids[mask]
    correct = 0
    for i in range(len(Xs)):
        tr = np.arange(len(Xs)) != i
        cents = {a: Xs[tr][ys[tr] == a].mean(axis=0)
                 for a in np.unique(ys[tr])}
        pred = min(cents, key=lambda a: np.sum((Xs[i] - cents[a]) ** 2))
        correct += int(pred == ys[i])
    acc = correct / max(len(Xs), 1)

    null = []
    for _ in range(200):
        yp = rng.permutation(ys)
        c = 0
        for i in range(len(Xs)):
            tr = np.arange(len(Xs)) != i
            cents = {a: Xs[tr][yp[tr] == a].mean(axis=0)
                     for a in np.unique(yp[tr])}
            pred = min(cents, key=lambda a: np.sum((Xs[i] - cents[a]) ** 2))
            c += int(pred == yp[i])
        null.append(c / max(len(Xs), 1))
    null = np.array(null)
    pval = float((np.sum(null >= acc) + 1) / (len(null) + 1))
    return dict(
        auc_same_vs_diff=float(auc_same),
        speakerid_acc=float(acc), speakerid_chance=1.0 / max(len(keep_animals), 1),
        speakerid_n_animals=len(keep_animals), speakerid_n_calls=int(mask.sum()),
        speakerid_perm_p=pval, speakerid_null_mean=float(null.mean()),
    )


# ------------------------------------------------- B/C. pseudo-dyad harness
def make_pairs(df):
    counts = df[df["phase"] == "alone"].groupby("animal_id").size()
    residents = [a for a, n in counts.items() if n >= MIN_RES_CALLS]
    partners = [a for a, n in counts.items() if n >= MIN_PART_CALLS]
    pairs = [(a, b) for a in residents for b in partners if a != b]
    return pairs, counts


def build_pseudo_sessions(df, X, pairs, tracks):
    """Feature-level pseudo-dyads from clean per-call features."""
    sessions, truth = [], {}
    alone = df[df["phase"] == "alone"]
    for a, b in pairs:
        ga = alone[alone["animal_id"] == a].sort_values("start_s")
        gb = alone[alone["animal_id"] == b].sort_values("start_s")
        half = len(ga) // 2
        anchor_idx = ga.index.to_numpy()[:half]
        test_a = ga.index.to_numpy()[half:]
        test_b = gb.index.to_numpy()
        idx = np.concatenate([test_a, test_b])
        times = df.loc[idx, "start_s"].to_numpy()
        posR = np.stack([A.interp_track(tracks[a], times)[k] for k in ("m1_x", "m1_y")], 1)
        posP = np.stack([A.interp_track(tracks[b], times)[k] for k in ("m1_x", "m1_y")], 1)
        key = f"{a}_vs_{b}"
        sessions.append(CORE.Session(
            key=key, stim_key=key, X=X[idx], times=times,
            level_db=df.loc[idx, "rf_amp_peak_db"].to_numpy(),
            pos_R=posR, pos_P=posP, anchor_X=X[anchor_idx],
        ))
        truth[key] = np.concatenate([np.ones(len(test_a)), np.zeros(len(test_b))])
    return sessions, truth


def eval_sessions(res, sessions, truth, tag):
    rows, call_rows = [], []
    for s in sessions:
        y = truth[s.key]
        r = res[s.key]
        if len(np.unique(y)) < 2:
            continue
        row = {"pair": s.key, "tag": tag, "n_res": int(y.sum()),
               "n_part": int((1 - y).sum())}
        for kind in ("voice", "full", "hmm"):
            p = r[f"p_res_{kind}"]
            row[f"auc_{kind}"] = float(roc_auc_score(y, p))
            row[f"acc_{kind}"] = float(np.mean((p > 0.5) == y))
        row["pi_hat"] = float(r["pi"])
        row["pi_true"] = float(y.mean())
        rows.append(row)
        for yy, pv, pf, ph, t in zip(y, r["p_res_voice"], r["p_res_full"],
                                     r["p_res_hmm"], s.times):
            call_rows.append({"pair": s.key, "tag": tag, "truth_res": int(yy),
                              "p_voice": pv, "p_full": pf, "p_hmm": ph,
                              "t": t})
    return rows, call_rows


def collision_rate(df, pairs):
    alone = df[df["phase"] == "alone"]
    tot, coll = 0, 0
    for a, b in pairs:
        ia = alone[alone["animal_id"] == a][["start_s", "end_s"]].to_numpy()
        ib = alone[alone["animal_id"] == b][["start_s", "end_s"]].to_numpy()
        for s0, s1 in ia:
            tot += 1
            if np.any((ib[:, 0] < s1) & (ib[:, 1] > s0)):
                coll += 1
    return coll / max(tot, 1)


# --------------------------------------------------- C. audio-level mixing
def audio_level_features(df, zcols, pairs_subset):
    """Re-extract features from summed waveforms; returns df-like with latents."""
    import torch
    from attr_20_train_embedder import ConvAE

    out = A.ensure_out()
    model = ConvAE()
    model.load_state_dict(torch.load(out / "embedder_scratch.pt",
                                     weights_only=True))
    model.eval()

    calls = A.load_calls()
    calls = calls[calls["phase"] == "alone"]
    frames = []
    for a, b in pairs_subset:
        for src, other in ((a, b), (b, a)):
            rows, crops, ids = EX.process_animal(src, calls, mix_with=other)
            if not rows:
                continue
            fd = pd.DataFrame(rows)
            with torch.no_grad():
                Z = model.encode(torch.from_numpy(
                    np.stack(crops).astype(np.float32)[:, None])).numpy()
            for i, z in enumerate(zcols):
                fd[z] = Z[:, i]
            fd["mix_pair"] = f"{a}_vs_{b}"
            fd["mix_role"] = "res" if src == a else "part"
            frames.append(fd)
        print(f"  mixed pair {a}x{b} done", flush=True)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------- D. NC
def negative_control(df, X, imodel):
    alone = df[df["phase"] == "alone"]
    sessions = []
    held = {}
    for animal, g in df.groupby("animal_id"):
        part = g[g["phase"] == "partner"]
        anch = g[g["phase"] == "alone"].sort_values("start_s")
        if len(anch) < 4:
            continue
        half = len(anch) // 2
        a_idx = anch.index.to_numpy()[:half]
        h_idx = anch.index.to_numpy()[half:]
        idxX = part.index.to_numpy()
        stim = A.STIM_MAP.get(animal) or f"solo_{animal}"
        sessions.append(CORE.Session(
            key=animal, stim_key=stim, X=X[idxX],
            times=part["start_s"].to_numpy(),
            level_db=part["rf_amp_peak_db"].to_numpy(),
            pos_R=part[["m1_x", "m1_y"]].to_numpy(),
            pos_P=part[["m2_x", "m2_y"]].to_numpy(),
            anchor_X=X[a_idx],
        ))
        held[animal] = h_idx
    res = CORE.fit_attribution(sessions, imodel, seed=SEED)
    rows = []
    for s in sessions:
        h = held[s.key]
        g = df.loc[h]
        p = score_with_model(
            res[s.key], X[h], g["start_s"].to_numpy(),
            g["rf_amp_peak_db"].to_numpy(),
            g[["m1_x", "m1_y"]].to_numpy(),
            np.full((len(h), 2), np.nan), imodel, res["_globals"])
        for pi_, cid in zip(p, g["call_id"]):
            rows.append({"animal_id": s.key, "call_id": cid, "p_res": float(pi_),
                         "genotype": g["genotype"].iloc[0]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- main
def main():
    out = A.ensure_out()
    df, zcols = load_all()
    X, transform, imodel = fit_spaces(df, zcols)
    summary = {"intensity_r2_loao": imodel.r2_loao,
               "intensity_enabled": bool(imodel.enabled)}

    print("A) identity signal...")
    summary.update(identity_signal(df, X))
    print(json.dumps({k: v for k, v in summary.items()}, indent=1))

    print("B) feature-level pseudo-dyads...")
    pairs, counts = make_pairs(df)
    print(f"   {len(pairs)} ordered pairs "
          f"({len(set(a for a, _ in pairs))} residents)")
    tracks = {a: A.load_tracking(a)
              for a in sorted({x for p in pairs for x in p})}
    sessions, truth = build_pseudo_sessions(df, X, pairs, tracks)
    res = CORE.fit_attribution(sessions, imodel, seed=SEED)
    rows_b, calls_b = eval_sessions(res, sessions, truth, "feature")
    summary["collision_rate"] = collision_rate(df, pairs)

    print("C) audio-level mixtures...")
    pair_scores = sorted(
        pairs, key=lambda p: min(counts[p[0]], counts[p[1]]), reverse=True)
    seen, subset = set(), []
    for a, b in pair_scores:
        if (b, a) in seen or (a, b) in seen:
            continue
        seen.add((a, b))
        subset.append((a, b))
        if len(subset) >= N_AUDIO_PAIRS:
            break
    mixed = audio_level_features(df, zcols, subset)
    Xm = transform(mixed, mixed[zcols].to_numpy())
    sessions_c, truth_c = [], {}
    for key, g in mixed.groupby("mix_pair"):
        a, b = key.split("_vs_")
        ga = df[(df["animal_id"] == a) & (df["phase"] == "alone")].sort_values("start_s")
        half = len(ga) // 2
        anchor_idx = ga.index.to_numpy()[:half]
        anchor_start = set(ga["start_s"].to_numpy()[:half].round(4))
        gres = g[g["mix_role"] == "res"]
        keep = ~gres["start_s"].round(4).isin(anchor_start)
        gtest = pd.concat([gres[keep], g[g["mix_role"] == "part"]])
        rowsX = Xm[[mixed.index.get_loc(i) for i in gtest.index]]
        times = gtest["start_s"].to_numpy()
        posR = np.stack([A.interp_track(tracks[a], times)[k] for k in ("m1_x", "m1_y")], 1)
        posP = np.stack([A.interp_track(tracks[b], times)[k] for k in ("m1_x", "m1_y")], 1)
        sessions_c.append(CORE.Session(
            key=key, stim_key=key, X=rowsX, times=times,
            level_db=gtest["rf_amp_peak_db"].to_numpy(),
            pos_R=posR, pos_P=posP,
            anchor_X=X[anchor_idx],
        ))
        truth_c[key] = (gtest["mix_role"] == "res").to_numpy().astype(float)
    res_c = CORE.fit_attribution(sessions_c, imodel, seed=SEED)
    rows_c, calls_c = eval_sessions(res_c, sessions_c, truth_c, "audio")

    print("D) negative control...")
    nc = negative_control(df, X, imodel)
    nc.to_csv(out / "benchmark_negative_control.csv", index=False)
    summary["negative_control_frac_res"] = float((nc["p_res"] > 0.5).mean())
    summary["negative_control_n"] = int(len(nc))

    bench_pairs = pd.DataFrame(rows_b + rows_c)
    bench_calls = pd.DataFrame(calls_b + calls_c)
    bench_pairs.to_csv(out / "benchmark_pairs.csv", index=False)
    bench_calls.to_csv(out / "benchmark_calls.csv", index=False)

    for tag in ("feature", "audio"):
        sub = bench_calls[bench_calls["tag"] == tag]
        if not len(sub):
            continue
        y, p = sub["truth_res"].to_numpy(), sub["p_hmm"].to_numpy()
        summary[f"{tag}_pooled_auc_hmm"] = float(roc_auc_score(y, p))
        summary[f"{tag}_err_res_at_05"] = float(np.mean(p[y == 1] <= 0.5))
        summary[f"{tag}_err_part_at_05"] = float(np.mean(p[y == 0] > 0.5))
        pairsub = bench_pairs[bench_pairs["tag"] == tag]
        summary[f"{tag}_mean_auc_hmm"] = float(pairsub["auc_hmm"].mean())
        summary[f"{tag}_mean_auc_voice"] = float(pairsub["auc_voice"].mean())
        summary[f"{tag}_mean_auc_full"] = float(pairsub["auc_full"].mean())
        summary[f"{tag}_n_pairs"] = int(len(pairsub))

    yb = bench_calls["truth_res"].to_numpy()
    pb = bench_calls["p_hmm"].to_numpy()
    fpr, tpr, _ = roc_curve(yb, pb)
    bins = np.linspace(0, 1, 11)
    binid = np.digitize(pb, bins) - 1
    calib = np.array([[bins[i] + 0.05, yb[binid == i].mean(), (binid == i).sum()]
                      for i in range(10) if (binid == i).sum() >= 5])
    np.savez_compressed(out / "benchmark_curves.npz",
                        roc_fpr=fpr, roc_tpr=tpr, calib=calib)

    with open(out / "benchmark_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
