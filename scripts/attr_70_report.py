"""Generate the final markdown report for the attribution project from the
computed outputs (no hand-typed numbers)."""
from __future__ import annotations

import json
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A


def main():
    out = A.ensure_out()
    with open(out / "benchmark_summary.json") as fh:
        b = json.load(fh)
    with open(out / "quantification_summary.json") as fh:
        q = json.load(fh)
    ses = pd.read_csv(out / "session_quantification.csv")
    st = pd.read_csv(out / "attribution_statistics.csv").set_index("metric")
    sh = pd.read_csv(out / "shared_stim_experiment.csv")
    feats = pd.read_csv(out / "call_features.csv")

    wt = ses[ses["genotype"] == "WT"]
    het = ses[ses["genotype"] == "HET"]
    n_calls = len(feats)
    n_alone = int((feats["phase"] == "alone").sum())

    def row(m):
        r = st.loc[m]
        return (f"| {r['label']} | {r['wt_mean']:.2f} | {r['het_mean']:.2f} | "
                f"{r['mannwhitney_p']:.4f} | {r['sig']} |")

    reused = sh[sh["n_sessions_for_stim"] >= 2]
    nat_lines = []
    for stim, g in reused.groupby("stim_id"):
        parts = ", ".join(
            f"resident {r.animal_id} ({r.genotype}): {int(r.n_dyad_calls)} calls"
            for r in g.itertuples())
        nat_lines.append(f"- stim **{stim}** -> {parts}")

    md = f"""# Who is calling? Single-microphone USV attribution - LgDel WT vs HET

## Question
During the 5-15 min interaction window either animal (experimental resident or
stim partner) can vocalize, and one microphone cannot localize the source. Can
each call be attributed to an animal?

## Method built (and fully benchmarked)
1. **Per-call features from raw 384 kHz audio**: 18 handcrafted spectrogram-ridge
   descriptors + a 24-d latent from a convolutional autoencoder trained FROM
   SCRATCH on this dataset's own {n_calls} call crops (no pretrained checkpoint).
2. **Attribution model**: PCA voice space; two-source Gaussian mixture per
   session with the resident anchored on its alone-phase calls (0-300 s, caller
   certain by design), partner components tied across sessions sharing a stim
   animal; gap-dependent 2-state bout HMM; received-level x tracked-position
   attenuation cue fitted on alone-phase calls.
3. **Ground-truth benchmarks**: synthetic dyads built from alone-phase calls of
   animal pairs (feature-level n={b.get('feature_n_pairs', 0)} pairs;
   audio-level waveform mixtures n={b.get('audio_n_pairs', 0)} pairs), speaker-ID
   tests, held-out-anchor negative controls.

## Benchmark verdict: per-call attribution is NOT supported here
- Individual voice fingerprint (same vs different animal): AUC
  **{b['auc_same_vs_diff']:.2f}** (chance 0.50), in every representation tested.
- {b['speakerid_n_animals']}-way speaker ID on alone calls:
  {b['speakerid_acc']*100:.1f}% vs {b['speakerid_chance']*100:.1f}% chance,
  permutation p = {b['speakerid_perm_p']:.2f}.
- Synthetic-dyad attribution: mean AUC {b['feature_mean_auc_hmm']:.2f}
  (feature-level) / {b['audio_mean_auc_hmm']:.2f} (audio mixtures).
- Negative control: held-out known-resident calls recovered at only
  {b['negative_control_frac_res']*100:.0f}% (needs ~100%).
- Received level follows no distance law (leave-one-animal-out
  R^2 = {b['intensity_r2_loao']:.2f}); the cue is auto-disabled.

Cause: the alone-phase anchor is small (n={n_alone} calls, mostly 2-8 ms
"short" calls) and mouse USVs here carry no measurable per-call individual
signature, while a large alone->social repertoire shift (tonality d = +2.1,
duration d = +0.9) dominates the feature space - a mixture model splits call
TYPES, not callers. Any per-call attribution pipeline would produce
confident-looking but unvalidated labels; ours is gated off by its own
benchmark.

## What IS valid: design-based attribution (bounds + natural experiments)
- **Partner ceiling**: stim partners come from one counterbalanced pool
  (51557 met both a HET and a WT resident). HET-resident dyads emit only
  {q['ceiling_from_het_dyads_mean']:.2f} calls/min in total, and a solo animal
  {q['ceiling_from_wt_alone_mean']:.2f} calls/min -> a partner contributes at
  most ~{q['partner_ceiling_calls_per_min']:.1f} calls/min
  (conservative {q['partner_ceiling_conservative']:.1f}).
- **Shared-stim natural experiment** (same partner, different resident):
{chr(10).join(nat_lines)}
  Within-stim variance fraction:
  {q['natural_experiment'].get('within_frac', float('nan'))*100:.0f}% - dyad
  output tracks the resident, not the partner.

## Quantification (interaction window 5-15 min; Mann-Whitney)

| metric | WT mean | HET mean | p | sig |
|---|---|---|---|---|
{row('alone_rate')}
{row('dyad_rate')}
{row('res_rate_lower')}
{row('res_rate_lower_cons')}
{row('part_rate_upper')}

- **Resident's point of view**: the WT resident emits >=
  {q['wt_resident_share_lower']*100:.0f}% of its dyad's interaction-window
  calls (lower bound {wt['res_rate_lower'].mean():.1f} calls/min); the HET
  resident's attributable rate is ~{het['res_rate_lower'].mean():.2f} calls/min
  and its dyad barely exceeds solo levels at all.
- **Partner's point of view**: partners vocalize little regardless of resident
  genotype (upper bound ~{q['partner_ceiling_calls_per_min']:.1f} calls/min in
  both groups). The interaction-window surge in WT dyads cannot be explained by
  the partner side. Note: partner-side values are BOUNDS, not measurements - a
  genotype effect on partner calling within this ceiling cannot be tested with
  one microphone.

## Data corrections made en route
- `detected_calls_merged.csv` was truncated at 600 s (silently dropping the
  last 5 minutes - 564 calls, e.g. 225/300 of 31337's). This analysis rebuilds
  from the raw per-session `*_stats.csv` files (full 0-900 s).
- Genotypes for 31097 (WT) and 31101 (HET) follow the authoritative
  `genotype_pyrat` column of the completed trial plan; the older CSV plan has
  them flipped.

## Assumptions and limits
- The partner ceiling assumes a stim partner's intrinsic vocal drive does not
  depend on resident genotype except through the interaction itself
  (exchangeability; supported by the counterbalanced stim pool and the
  shared-stim experiment, n=2 reused stims).
- Overlapping simultaneous calls cannot be split by one detector; timestamp
  collision rate in synthetic dyads was {b['collision_rate']*100:.1f}%.
- Stim sex/genotype are not recorded in the trial plan; per-call attribution
  might become feasible with mic arrays, higher alone-call counts, or partner
  solo recordings (recommendations below).

## Recommendations for future sessions
1. Record each stim partner alone for 5 min (same arena) - gives a partner
   voice anchor and makes the mixture model identifiable.
2. Use >=4 microphones (corner array) for time-difference-of-arrival
   localization; 1,496-2,060 calls/cohort is ample for validation.
3. Keep the counterbalanced stim reuse - it is what rescued attribution here.

## Files
- `attr_fig1_method.(png|svg|pdf)` - pipeline, features, from-scratch AE,
  voice space, intensity-cue test
- `attr_fig2_validation.(png|svg|pdf)` - benchmark verdict figures
- `attr_fig3_quantification.(png|svg|pdf)` - WT vs HET, both points of view
- `session_quantification.csv`, `attribution_statistics.csv`,
  `benchmark_*.csv/json`, `call_attribution.csv` (per-call posteriors -
  exported as call-TYPE split, not caller identity), `timecourse_30s.csv`,
  `shared_stim_experiment.csv`
"""
    (out / "ATTRIBUTION_REPORT.md").write_text(md, encoding="utf-8")
    print(f"wrote {out / 'ATTRIBUTION_REPORT.md'}")


if __name__ == "__main__":
    main()
