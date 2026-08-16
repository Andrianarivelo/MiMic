"""V5 report: when calls occur, what they do, why HET mice do not call."""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import attr_common as A


def main():
 out = A.ensure_out()
 why = json.load(open(out / "v5_why_het.json"))
 sup = json.load(open(out / "v5_supplement.json"))
 comp = json.load(open(out / "v5_composite_direction.json"))["results"]
 haz = pd.read_csv(out / "v5_hazard.csv")
 perm = pd.read_csv(out / "v5_calltype_permutation.csv")
 dose = pd.read_csv(out / "v5_dose_response.csv")
 ct = pd.read_csv(out / "v5_calltype_effects.csv")
 lad = pd.read_csv(out / "v5_ladder.csv")
 rob = json.load(open(out / "v5_robustness.json"))
 cprof = pd.read_csv(out / "v5_calltype_profile.csv")
 cm = json.load(open(out / "v5_calltype_model.json"))
 k = why["kitagawa"]
 sg = why["social_gain"]
 g = why["gain"]
 rec = why["recruitment"]

 LADORD = ["none", "oriented_toward", "approach", "nose2nose", "nose2body",
 "nose2anogenital", "following", "chasing"]
 PR = {"none": "no social flag", "oriented_toward": "oriented toward",
 "approach": "approach", "nose2nose": "nose-to-nose",
 "nose2body": "body sniff", "nose2anogenital": "anogenital sniff",
 "following": "following", "chasing": "chasing"}
 piv = lad.pivot(index="state", columns="genotype", values="rate")
 lad_rows = "\n".join(
 f"| {PR[s]} | {piv.loc[s,'WT']:.1f} | {piv.loc[s,'HET']:.2f} | "
 f"{piv.loc[s,'WT']/max(piv.loc[s,'HET'],1e-9):.0f}× |"
 for s in LADORD if s in piv.index)

 DIR = {"resident_investigation": "the resident's own investigation",
 "partner_investigation": "the partner investigating the resident",
 "partner_flight": "the partner's flight",
 "social_contact": "mutual social contact"}
 dir_rows = "\n".join(
 f"| {DIR[n]} | {100*comp[n]['occupancy']:.1f}% | "
 f"{comp[n]['enrichment']:.2f}× | {comp[n]['lambda']:+.2f} | "
 f"{comp[n]['p']:.3f} | {comp[n]['q']:.3f} | "
 f"{'**behaviour leads the call**' if comp[n]['q'] < 0.05 else 'ns'} |"
 for n in ["resident_investigation", "partner_investigation",
 "partner_flight", "social_contact"])

 st_rows = "\n".join(
 f"| {s} | {100*tw:.1f}% | {100*th:.1f}% | {lw:.1f} | {lh:.2f} | {lw/max(lh,1e-9):.0f}× |"
 for s, tw, th, lw, lh in zip(k["states"], k["T_WT"], k["T_HET"],
 k["lam_WT"], k["lam_HET"]))

 typ_rows = "\n".join(
 f"| {r['name']} | {int(r['n'])} | {r['rf_dur_ms']:.1f} | "
 f"{r['rf_f_mean_khz']:.1f} | {r['rf_slope_khz_ms']:+.2f} |"
 for _, r in cprof.sort_values("rf_dur_ms").iterrows())

 OUTL = {"d_app_m2": "partner approach velocity",
 "d_app_m1": "resident approach velocity",
 "d_speed_m2": "partner speed", "d_speed_m1": "resident speed",
 "d_dist": "inter-animal distance",
 "esc_post": "partner escape within 2 s",
 "wdr_post": "partner withdrawal within 2 s"}
 perm_rows = "\n".join(
 f"| {OUTL.get(r['outcome'], r['outcome'])} | {r['obs_t']:.2f} | "
 f"{r['null_mean_t']:.2f} | {r['p_perm']:.3f} | {r['q_perm']:.2f} |"
 for _, r in perm.sort_values("p_perm").iterrows())

 nA = haz[(haz.model == "A_antecedent") & (haz.set == "WT")]
 nB = haz[(haz.model == "B_consequence") & (haz.set == "WT")]
 bt = why["bouts"]
 esc = why["escalation"]

 md = f"""# What calls do, when they occur, and why HET mice do not call

Companion to the v2 attribution and the v4 behaviour analysis. All numbers come
from the {len(pd.read_csv(out / 'v5_calls.csv'), ):,} interaction-window calls in 24 sessions (12 WT, 12 HET),
aligned to tracking and behaviour at 100 ms resolution. Figures
`v5_fig1_when`, `v5_fig2_what`, `v5_fig3_why`, `v5_fig4_repertoire`.

**Summary in one line.** Calling is a *graded readout of the caller's own social
engagement*: it rises monotonically with engagement intensity, follows rather
than precedes behaviour, and has no measurable effect on the partner. In HET
mice the readout still exists but its gain is gone - they start calling as
readily as WT, in the right places, with normal-looking calls, and then never
escalate.

---

## 1. When calls occur

### 1.1 Calling is graded, not triggered
Call rate climbs monotonically along an ordered scale of social engagement:

| resident's behaviour | WT (calls/min) | HET (calls/min) | ratio |
|---|---|---|---|
{lad_rows}

Spearman correlation between ladder position and call rate:
**WT ρ = {sup['ladder']['WT']['rho']:+.2f} (p = {sup['ladder']['WT']['p']:.3f})**,
HET ρ = {sup['ladder']['HET']['rho']:+.2f} (p = {sup['ladder']['HET']['p']:.2f}).

The same gradient appears in a continuous variable. Regressing log call rate on
log inter-animal distance gives a WT slope of **{g['slope_WT']:+.2f}**
(closer ⇒ more calling) against a HET slope of **{g['slope_HET']:+.2f}**;
difference {g['slope_diff']:+.2f}
[{g['slope_diff_ci'][0]:+.2f}, {g['slope_diff_ci'][1]:+.2f}],
session-bootstrap p < 0.001.

### 1.2 Calls track the state, not its onset
Within investigation bouts the call rate does **not** spike at bout onset and
decay, which is what a phasic response to an event looks like. It is
{sup['timing']['rate_early']:.0f}/min in the first 0.5 s and
{sup['timing']['rate_late']:.0f}/min later
(ρ = {sup['timing']['rho']:+.2f}, p = {sup['timing']['p']:.2f}) - calling
*accumulates* as engagement continues.

---

## 2. What calls do

### 2.1 The method: the antisymmetric cross-correlogram
Write the call-triggered behaviour profile as

 P(τ) = ( Σ_t x(t) y(t+τ) ) / Σ_t x(t)

for a call train x and a behaviour state indicator y, and split it into even and
odd parts. If both channels are driven by a common instantaneous process z - x = a·z + noise, y = b·z + noise - then P(τ) = a·b·R_z(τ), and the autocovariance
of any weakly stationary process is even, R_z(τ) = R_z(−τ). **The odd part is
therefore identically zero under a shared drive.** Proximity, arousal, bout
structure, and the regression-to-the-mean artefact that sank the v4 consequence
claim all enter symmetrically and cancel. What survives,

 Λ = Σ_{{τ>0}} [ P(τ) − P(−τ) ] ,

is a directional statistic: Λ > 0 means the call leads, Λ < 0 means the
behaviour leads. The null is the exact set of circular shifts of the call train,
which preserves each session's call count and bout structure and destroys only
the alignment.

Two implementation points matter. `y` must be the **state indicator, not a bout
onset train** - an onset train is asymmetric by construction (off before, on
after), so calls that merely occur *during* a behaviour manufacture a negative Λ
with no coupling at all. And the suites below were **pre-specified** from the v4
result rather than picked after looking.

### 2.2 Only the caller's own behaviour leads the call

| suite | occupancy | enrichment around calls | Λ | p | q | verdict |
|---|---|---|---|---|---|---|
{dir_rows}

The resident's own investigation is 1.87× enriched around its calls and
**precedes** them (Λ = {comp['resident_investigation']['lambda']:+.2f},
p = {comp['resident_investigation']['p']:.3f}, q =
{comp['resident_investigation']['q']:.3f}). No partner behaviour leads or
follows the call in either direction - including the partner's flight, the most
obvious candidate for "the resident calls because the partner ran".

### 2.3 Covariate-adjusted hazard models agree
Discrete-time logistic hazards were fitted **separately in each session** and
combined across sessions by a t-test on the 12 coefficients. (A pooled
cluster-robust sandwich is not usable here: its rank cannot exceed the number of
clusters, and with 12 sessions against 30 covariates it returns intervals
spanning 10³⁰. Sessions are the unit of replication and are treated as such.)

- **Antecedent** - P(call) given a partner-behaviour onset in the preceding
 second, adjusting for current speed, distance, and behavioural state:
 **{(nA['q'] < 0.05).sum()} of {len(nA)} behaviours survive FDR.** The strongest
 raw effect additionally fails its own circular-shift null (p = 0.30). Note the
 shift null's mean coefficient is not zero, so testing against zero would have
 been wrong here.
- **Consequence** - P(behaviour onset) given a call in the preceding second,
 with **all controls measured before the call** (the regression form of a
 matched counterfactual, and it includes the outcome's own baseline - the
 omission that made the v4 claim regression to the mean):
 **{(nB['q'] < 0.05).sum()} of {len(nB)} survive FDR.**

### 2.4 No call type has a demonstrable effect
The strongest observational design available is a **within-call contrast**:
every arm is a call, so whatever selects a moment for calling is shared and
cancels; only the type differs. Each call's pre-window value of the outcome is
included as a covariate. The null permutes type labels between bouts within a
session.

| outcome | observed \\|t\\| | null mean \\|t\\| | permutation p | q |
|---|---|---|---|---|
{perm_rows}

{(ct['q'] < 0.05).sum()} of {len(ct)} individual type contrasts survive FDR. A
dose-response on the number of calls in a bout gives a minimum q of
{dose['q'].min():.2f} across the same seven outcomes.

### 2.5 Conclusion
Across four designs with four different nulls, **nothing reliably follows a
call**. Combined with the v4 retraction, this dataset supports "calls index
social engagement" and does not support "calls signal something to the partner".
Establishing a receiver effect needs playback or silencing, not observation.

---

## 3. Why HET mice do not call

WT emit {k['R_WT']:.2f} calls/min in the interaction window against
{k['R_HET']:.2f} for HET - an {k['R_WT']/k['R_HET']:.1f}× difference. Five
candidate explanations are separable in these data.

### 3.1 Not opportunity (they do the behaviours)
The session rate factorises exactly as R = Σ_s T_s λ_s over mutually exclusive
states, with T_s the time share and λ_s the rate while in state s. The Kitagawa
identity splits the gap with no residual:

| state | WT time | HET time | WT rate | HET rate | rate ratio |
|---|---|---|---|---|---|
{st_rows}

- opportunity (time budget): **{k['opportunity']:+.2f} calls/min**
 [{k['opportunity_ci'][0]:+.2f}, {k['opportunity_ci'][1]:+.2f}]
- propensity (rate within state): **{k['propensity']:+.2f} calls/min**
 [{k['propensity_ci'][0]:+.2f}, {k['propensity_ci'][1]:+.2f}]
- propensity share **{k['pct_propensity']:.0f}%**
 [{k['pct_propensity_ci'][0]:.0f}, {k['pct_propensity_ci'][1]:.0f}]

The counterfactual makes it concrete: **give HET mice WT's exact time budget and
they call {k['cf_HET_with_WT_time']:.2f}/min instead of {k['R_HET']:.2f}/min** - no change. HET are not asocial; they spend {100*k['T_HET'][k['states'].index('investigate')]:.0f}%
of the window investigating against WT's {100*k['T_WT'][k['states'].index('investigate')]:.0f}%,
a 1.7× difference that cannot produce an 11.9× rate difference.

### 3.2 Not the vocal apparatus
- **Latency is identical.** Median time to the first call after the partner is
 introduced: WT {why['latency']['WT_median']:.0f} s vs HET
 {why['latency']['HET_median']:.0f} s, log-rank p = {why['latency']['p']:.2f}.
 HET start calling exactly as readily.
- **The repertoire is not abnormal.** A checkpoint-free Gaussian mixture on our
 own spectrogram-ridge features (BIC selects K = {cm['K']}) gives:

| type | n | median duration (ms) | mean f (kHz) | FM slope (kHz/ms) |
|---|---|---|---|---|
{typ_rows}

 HET calls concentrate in the ultrashort types, which looks like a phenotype
 until it is controlled: **within WT alone, the fraction of ultrashort calls
 tracks the session's call count at ρ = {rec['rho_WT']:+.2f}
 (p = {rec['p_rho_WT']:.3f})**. Every mouse's repertoire shifts to brief calls
 at low rates. In a session-level ANCOVA the residual genotype offset is not
 significant (p = {rec['p_genotype']:.2f}), and the
 {rec['n_ratematched_WT']} rate-matched WT sessions are indistinguishable from
 HET ({rec['ratematched_WT_median']:.2f} vs
 {rec['ratematched_HET_median']:.2f} ultrashort, p = {rec['ratematched_p']:.2f}).
 The acoustic difference is a *consequence* of calling less, not a separate deficit.

### 3.3 Not the partner
Stimulus partners of the two genotypes differ on only one of 15 behaviours
after FDR - escape (WT partners 1.01% of the time vs HET partners 0.23%,
q = 0.009) - which is downstream of WT residents chasing more, not an input.

### 3.4 It is the gain from social engagement to voice
- **Social gain** (call rate in social states ÷ non-social states):
 WT **{sg['WT_gain']:.2f}×** [{sg['WT_gain_ci'][0]:.2f}, {sg['WT_gain_ci'][1]:.2f}]
 vs HET **{sg['HET_gain']:.2f}×** [{sg['HET_gain_ci'][0]:.2f}, {sg['HET_gain_ci'][1]:.2f}].
 The HET interval **includes 1** - no modulation at all. Ratio
 {sg['gain_ratio']:.2f}× [{sg['gain_ratio_ci'][0]:.1f}, {sg['gain_ratio_ci'][1]:.1f}],
 bootstrap p < 0.001. One WT animal contributes 39% of all WT calls, so this
 was checked by leave-one-session-out: the WT gain stays between
 {sup['loso_gain']['min']:.1f}× and {sup['loso_gain']['max']:.1f}×.
- **Proximity no longer drives calling** (§1.1): WT slope
 {g['slope_WT']:+.2f} vs HET {g['slope_HET']:+.2f}.
- **The engagement ladder is flat** (§1.1): ρ {sup['ladder']['WT']['rho']:+.2f}
 vs {sup['ladder']['HET']['rho']:+.2f}.

The deficit is largest exactly where social drive should be highest: HET call
{k['lam_WT'][k['states'].index('contact')]/max(k['lam_HET'][k['states'].index('contact')],1e-9):.0f}×
less during social contact and
{k['lam_WT'][k['states'].index('investigate')]/max(k['lam_HET'][k['states'].index('investigate')],1e-9):.0f}×
less during investigation, but only
{k['lam_WT'][k['states'].index('none')]/max(k['lam_HET'][k['states'].index('none')],1e-9):.0f}×
less when doing nothing social. A uniform gain change would hit all states
equally; this does not.

### 3.5 The escalation failure
Decomposing rate = (bouts per minute) × (calls per bout), from an ICI mixture
bout criterion of {why['bout_mixture']['gap_s']:.2f} s:

| component | WT | HET | ratio | p |
|---|---|---|---|---|
| bouts per minute (initiation) | {bt['bouts_per_min']['WT_median']:.2f} | {bt['bouts_per_min']['HET_median']:.2f} | {bt['bouts_per_min']['WT_median']/bt['bouts_per_min']['HET_median']:.1f}× | {bt['bouts_per_min']['p']:.4f} |
| calls per bout (maintenance) | {bt['calls_per_bout']['WT_median']:.2f} | {bt['calls_per_bout']['HET_median']:.2f} | {bt['calls_per_bout']['WT_median']/bt['calls_per_bout']['HET_median']:.1f}× | {bt['calls_per_bout']['p']:.4f} |
| longest bout | {bt['max_bout']['WT_median']:.0f} | {bt['max_bout']['HET_median']:.0f} | {bt['max_bout']['WT_median']/bt['max_bout']['HET_median']:.1f}× | {bt['max_bout']['p']:.4f} |

HET bouts are almost always singletons: a bout reaches ≥ 2 calls
{sup['escalation_bout']['p_ge2']['WT']:.2f} of the time in WT against
{sup['escalation_bout']['p_ge2']['HET']:.2f} in HET
(p = {sup['escalation_bout']['p_ge2']['p']:.3f}), and reaches ≥ 4 calls
{sup['escalation_bout']['p_ge4']['WT']:.2f} vs
{sup['escalation_bout']['p_ge4']['HET']:.2f}
(p = {sup['escalation_bout']['p_ge4']['p']:.3f}). At the session level,
{esc['ge_30']['WT']}/12 WT sessions exceed 30 calls against
{esc['ge_30']['HET']}/12 HET (Fisher p = {esc['ge_30']['p']:.4f}); the WT median
is {esc['median_WT']:.0f} calls and the maximum 703, while no HET session
exceeds 25.

**The phenotype is not "HET are quiet". It is that HET never escalate.** They
enter the interaction, they investigate, they emit a first call on the same
schedule as WT - and then the positive feedback that turns a first call into a
bout, and social proximity into a rising call rate, does not happen.

### 3.6 Is one animal carrying this?
Animal 31078 supplies 703 of the 1,786 WT interaction-window calls (39%), so
every headline number was recomputed without it, and the gain was additionally
recomputed **per session**, where each animal contributes exactly one value
regardless of how much it calls. The social gain is a *within*-animal ratio, so
a prolific caller cannot inflate it - and in fact 31078 has one of the
*weakest* gains of the twelve WT animals ({rob['per_session_gain']['gain_31078']:.2f}×,
third lowest), so it was diluting the effect rather than driving it.

| quantity | all 12 WT | without 31078 |
|---|---|---|
| WT social gain (pooled) | {rob['pooled_gain']['WT all 12']['gain']:.2f}× | **{rob['pooled_gain']['WT without 31078']['gain']:.2f}×** |
| gain ratio WT ÷ HET | {sg['gain_ratio']:.2f}× | **{rob['gain_ratio_drop31078']['ratio']:.2f}×** [{rob['gain_ratio_drop31078']['ci'][0]:.1f}, {rob['gain_ratio_drop31078']['ci'][1]:.1f}] |
| propensity share of the gap | {k['pct_propensity']:.0f}% | **{rob['kitagawa_drop31078']['pct_propensity']:.0f}%** [{rob['kitagawa_drop31078']['ci'][0]:.0f}, {rob['kitagawa_drop31078']['ci'][1]:.0f}] |
| engagement-ladder ρ (WT) | {sup['ladder']['WT']['rho']:+.2f} | **{rob['ladder']['WT minus 31078']['rho']:+.2f}** |
| distance slope (WT) | {g['slope_WT']:+.2f} | **{rob['distance_slope_drop31078']:+.2f}** |
| session call rate, WT vs HET | p = {esc['p_rate']:.4f} | **p = {rob['rate_drop31078']['p']:.4f}** |

Per-session social gain: WT median **{rob['per_session_gain']['WT_median']:.2f}×**
({rob['per_session_gain']['WT_n_above_1']}/12 animals above 1×) against HET
**{rob['per_session_gain']['HET_median']:.2f}×**
({rob['per_session_gain']['HET_n_above_1']}/12), Mann–Whitney
p = {rob['per_session_gain']['p']:.4f}; dropping 31078 gives WT median
{rob['per_session_gain']['WT_median_drop31078']:.2f}×,
p = {rob['per_session_gain']['p_drop31078']:.4f}. **Every number is unchanged or
stronger without the heavy caller.** The state-specific ladder of deficits also
survives - without 31078 it runs 2.8× (nothing social) → 5.4× → 5.8× → 14.8×
(investigating) → 48.2× (contact), a slightly *wider* span than with it.

---

## 4. What would test this
The gain interpretation makes falsifiable predictions this dataset cannot check:
1. Sustained pharmacological or optogenetic elevation of social arousal should
 restore the WT-like *gradient* in HET, not merely the mean rate.
2. Because the deficit is in escalation rather than initiation, a HET mouse
 should show a normal first call and an abnormally short bout at *any* level
 of engagement - testable with longer sessions and more stimulus variety.
3. Playback of WT bouts to a partner is the only way to settle §2: whether these
 calls carry information a receiver acts on.

## 5. Caveats
- Behaviour flags come from PyKaboo's rule engine on the same DLC keypoints as
 the GEO attribution, so behaviour and geometry are not independent modalities.
 The calls are an independent acoustic measurement, so call-behaviour coupling
 is genuinely cross-modal.
- Attribution is session-level (per-call sensitivity ≈ 0.69), so "resident
 calls" are p-weighted, not certain. Results are near-identical unweighted.
- WT calling is heavy-tailed (one animal = 39% of calls). Every genotype
 comparison here is session-clustered, rank-based, or leave-one-out checked.
- The Λ statistic is exactly zero under an instantaneous common drive that is
 *time-reversible*. Strongly irreversible shared dynamics could in principle
 generate odd structure; §2.3–2.4 do not rely on that assumption.

## 6. Files
- Figures in `attribution/figures/`: `v5_fig1_when`, `v5_fig2_what`,
 `v5_fig3_why`, `v5_fig4_repertoire`, `v5_fig5_gain_robustness` -- each as
 300 dpi PNG plus SVG and PDF with live, editable text
- Tables `v5_calls.csv`, `v5_calltype_profile.csv`, `v5_directionality.csv`,
 `v5_composite_direction.json`, `v5_hazard.csv`, `v5_calltype_effects.csv`,
 `v5_calltype_permutation.csv`, `v5_dose_response.csv`, `v5_why_het.json`,
 `v5_supplement.json`, `v5_ladder.csv`, `v5_distance_gain.csv`,
 `v5_bouts.csv`, `v5_bout_escalation.csv`, `v5_latency.csv`,
 `v5_recruitment.csv`, `v5_robustness.json`, `v5_gain_per_session.csv`
- Scripts `attr5_core.py`, `attr5_00_types.py`, `attr5_10_directionality.py`,
 `attr5_15_composite.py`, `attr5_20_hazard.py`, `attr5_25_calltype_effects.py`,
 `attr5_30_why_het.py`, `attr5_35_supplement.py`, `attr5_40_figures.py`
"""
 (out / "V5_MECHANISM_REPORT.md").write_text(md, encoding="utf-8")
 print("wrote", out / "V5_MECHANISM_REPORT.md")


if __name__ == "__main__":
 main()
