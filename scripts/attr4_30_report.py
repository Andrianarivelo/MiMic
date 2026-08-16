"""Report for the call x behaviour analysis."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A
from attr4_20_figures import PRETTY, pretty


def main():
    out = A.ensure_out()
    enr = pd.read_csv(out / "beh_enrichment.csv")
    cst = pd.read_csv(out / "beh_consequence_stats.csv")
    pred = json.load(open(out / "beh_prediction.json"))
    val = json.load(open(out / "beh_validation.json"))
    onset = pd.read_csv(out / "beh_peri_onset.csv")

    m1 = enr[enr["actor"] == "m1"].sort_values("log2_RR_adj", ascending=False)
    sig = m1[m1["q_adj"] < 0.05]

    def row(r):
        return (f"| {PRETTY.get(r['behavior'], r['behavior'])} | "
                f"{r['occupancy_pct']:.1f}% | {int(r['calls_in_state'])} | "
                f"{2**r['log2_RR_crude']:.2f}× | {2**r['log2_RR_adj']:.2f}× | "
                f"{r['q_adj']:.3f} | {int(r['n_sessions_positive'])}/"
                f"{int(r['n_sessions'])} |")

    top_lines = "\n".join(row(r) for _, r in m1.iterrows())

    cres = cst[(cst["role"] == "res") & (cst["window"] == "early_0.2-1s")]
    cres = cres.sort_values("p_boot")

    def crow(r):
        return (f"| {pretty(r['metric'])} | {r['pooled_diff']:+.3g} | "
                f"[{r['boot_lo']:+.3g}, {r['boot_hi']:+.3g}] | "
                f"{r['p_boot']:.3f} | {r['q_boot']:.3f} |")
    cons_lines = "\n".join(crow(r) for _, r in cres.head(8).iterrows())
    asym = val.get("actor_asymmetry", {})
    per_b = asym.get("per_behavior", {})
    asym_lines = "\n".join(
        f"| {PRETTY.get(k, k)} | {v:+.2f} |" for k, v in
        sorted(per_b.items(), key=lambda kv: -kv[1]))
    n_pos = sum(1 for v in per_b.values() if v > 0)
    diag = json.load(open(out / "beh_diag_summary.json"))
    rtd = pd.read_csv(out / "beh_diag_retest.csv")
    rtd = rtd[rtd["metric"] == "app_m2"]
    retest_lines = "\n".join(
        f"| {r['matching']} | {r['subset']} | {int(r['n_calls'])} | "
        f"{r['pooled_diff']:+.1f} | [{r['boot_lo']:+.1f}, {r['boot_hi']:+.1f}] | "
        f"{'**yes**' if r['sig'] else 'no'} |" for _, r in rtd.iterrows())

    coef = pred["coefficients_both"]
    ei = enr.set_index("flag")
    coef = {k: v for k, v in coef.items()
            if k not in ei.index or ei.loc[k, "calls_in_state"] >= 30}
    top_coef = sorted(coef.items(), key=lambda kv: -abs(kv[1]))[:10]
    coef_lines = "\n".join(f"| {pretty(k)} | {v:+.2f} |" for k, v in top_coef)

    het = val["calltype_heterogeneity"]
    het_lines = "\n".join(
        f"- **{k}** types (n={v['n_types']}, {v['n_calls']} calls): "
        f"heterogeneity p = {v['p']:.3f}" for k, v in het.items())

    md = f"""# Behavioural context of ultrasonic calls

{val['n_calls_partner']:,} interaction-window calls across {val['n_sessions']} sessions, aligned to the
per-frame behaviour stream at 100 ms resolution.

## Important caveat about the labels
The behaviour flags come from PyKaboo's live **rule engine**
(`behavior_backend = "rules"`), computed from the same DLC keypoints that the
v2 GEO attribution uses. They are therefore a different formalisation of the
same tracking stream, **not an independent modality**: agreement between calls
and "chasing" is not independent confirmation of the geometry model. The
calls themselves are an independent acoustic measurement, so call-behaviour
coupling is a genuine cross-modal result.

Two confounds are controlled throughout: calls favour movement (shown in the
v3 noise control) and behaviours differ in speed and proximity. Every rate
ratio is therefore reported both raw and **adjusted by stratifying on speed x
inter-animal distance terciles** (Mantel-Haenszel person-time estimator), and
significance comes from **circular-shift nulls** that preserve call bout
structure exactly.

## A. Where calls happen (resident's own behaviour)

| behaviour | % of time | calls | raw rate ratio | adjusted | q | sessions agreeing |
|---|---|---|---|---|---|---|
{top_lines}

Adjustment matters: raw ratios are roughly twice the adjusted ones, so about
half of the apparent enrichment is simply that these behaviours happen while
the animals are moving and close together - and about half is specific to the
behaviour itself.

{len(sig)} of {len(m1)} rows survive FDR correction. The profile replicates
across animals: split-half Spearman rho = **{val['splithalf_rho_mean']:.2f}**
(95% CI {val['splithalf_rho_lo']:.2f}-{val['splithalf_rho_hi']:.2f}).

### Actor versus target
For directional behaviours the flag marks who is *doing* it. Enrichment
follows the actor: the resident's own investigation is associated with calls,
being investigated by the partner is not.

| behaviour | log2 (resident acting − partner acting) |
|---|---|
{asym_lines}

All {n_pos} of {len(per_b)} directional behaviours point the same way, and the
mean gap is **{asym.get('stat_log2', float('nan')):+.2f} log2
({2**asym.get('stat_log2', 0):.1f}x)** - but against the circular-shift null
this single summary test gives **p = {asym.get('p', float('nan')):.2f}**, so
the asymmetry is suggestive rather than established. Note also that these
labels derive from the same keypoints as the GEO attribution, so this is
consistent with the pursuit-geometry result rather than independent support
for it.

## B. When calls happen
`beh_peri_call.csv` gives behaviour probability in a +-5 s window around every
call; `beh_peri_onset.csv` gives call rate around behaviour-bout onsets. Both
carry 95% bands from the circular-shift null (figure `v4_fig2_when`).

## C. What follows a call — RETRACTED (regression to the mean)

An earlier version of this report claimed that after a call the caller slows
and the partner stops fleeing and approaches. **That claim does not survive
scrutiny and is withdrawn.** The diagnostic is in
`v4_fig5_consequence_retracted` and `beh_diag_*.csv`.

Why it fails:

1. **Calls are emitted at kinematic extremes.** While the resident is
   anogenital-sniffing, the partner is retreating at
   {diag['app_m2_during_anogenital']:.0f} px/s, versus {diag['app_m2_elsewhere']:.0f} px/s
   elsewhere; during following and chasing it is near -290 px/s. A moment
   sampled at an extreme must relax towards the mean afterwards whatever the
   call does.
2. **The matching failed on the outcome itself.** Controls were matched on
   behaviour state, speed and distance, but not on the partner's approach
   velocity. Pre-call balance is poor (standardised mean difference
   {diag['balance']['app_m2']:+.2f} for partner approach velocity,
   {diag['balance']['speed_m2']:+.2f} for partner speed), so calls started
   from systematically more extreme states than their controls.
3. **The effect disappears under every stricter test.** Matching additionally
   on pre-call approach velocity and partner speed, or restricting to isolated
   calls (no other call within 1 s), or to bout-onset calls, all drive the
   partner-approach effect to zero:

| matching | call subset | n | effect (px/s) | 95% CI | survives |
|---|---|---|---|---|---|
{retest_lines}

4. **The raw traces overlap.** Plotted as absolute values rather than
   differences, call moments and matched control moments follow the same
   trajectory after the event; they differ only in where their baselines sat.

**Honest conclusion: this dataset shows WHERE calls occur, but does not
demonstrate a measurable behavioural consequence of calling.** Establishing
one needs an intervention (playback or silencing), not observational contrast.

## C-old. Original matched-counterfactual output (superseded)
Each call is compared with no-call moments **from the same session matched on
behavioural state, speed quintile and distance tercile**, contrasting the
change from the second before the call to the 0.2-2 s after it. Confidence
intervals come from a cluster bootstrap over sessions.

Window 0.2-1 s after the call (kinematics in px/s, behaviour flags as
probabilities):

| outcome | call − matched control | 95% CI | p | q |
|---|---|---|---|---|
{cons_lines}

These numbers are kept only for the record. **Section C above shows they are
an artefact of unbalanced baselines; do not cite them.**

## D. What correlates best with calling
Leave-one-session-out prediction of which 100 ms bins contain a call
({pred['n_call_bins']:,} call bins out of {pred['n_bins']:,}):

| feature set | held-out AUC |
|---|---|
| kinematics only | {pred['kinematics']['loso_auc_mean']:.3f} ± {pred['kinematics']['loso_auc_sd']:.3f} |
| behaviour only | {pred['behaviour']['loso_auc_mean']:.3f} ± {pred['behaviour']['loso_auc_sd']:.3f} |
| both | {pred['both']['loso_auc_mean']:.3f} ± {pred['both']['loso_auc_sd']:.3f} |

Strongest multivariate coefficients:

| feature | weight |
|---|---|
{coef_lines}

Behaviour and kinematics are individually about equally informative and only
partly redundant, but all three models sit near AUC 0.6: knowing exactly what
the animals are doing still leaves most of the moment-to-moment timing of
calling unexplained. Calling is coupled to social context, not dictated by it.

## E. Do call types differ in context?
{het_lines}

Syllable classes come from the upstream VocalPy classifier and are used here
only as labels; the duration terciles are measured from our own spectrogram
ridges and are checkpoint-free.

## Files
- Figures: `v4_fig1_where`, `v4_fig2_when`, `v4_fig3_consequences`,
  `v4_fig4_calltypes` (png/svg/pdf each)
- Tables: `beh_enrichment.csv`, `beh_peri_call.csv`, `beh_peri_onset.csv`,
  `beh_consequence.csv`, `beh_consequence_stats.csv`, `beh_calltype.csv`,
  `beh_prediction.json`, `beh_validation.json`, `beh_bins.csv.gz`
"""
    (out / "V4_BEHAVIOR_REPORT.md").write_text(md, encoding="utf-8")
    print("wrote", out / "V4_BEHAVIOR_REPORT.md")


if __name__ == "__main__":
    main()
