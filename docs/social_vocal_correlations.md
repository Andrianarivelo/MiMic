# Social behavior and dyad vocalization correlations

Run `python scripts/social_vocal_correlations.py` in the available scientific
Python environment, open the script in an IDE, or double-click the adjacent
`.cmd` launcher. All numeric parameters are at the top of that script;
figure styling is in `scripts/social_vocal_correlation_figures.py`.

The workflow reads original detector CSVs and the completed trial plan without
modifying raw data or metadata. It uses the 300-900 s partner window and exact
shared observed-frame exposure. Counts and acoustic features concern dyad calls,
without geometrically assigning the caller. Vocal durations are clipped to
observed intervals; missing tracking breaks both behavior episodes and vocal
bouts. Overlapping behavioral actor flags are unioned before dyad measurements.

The fixed primary exploration contains 13 social behaviors, three behavioral
metrics (cumulative active duration, mean active episode duration and episode
rate), and eight vocal metrics. One 312-test family receives BH, BY and Holm
adjustments. Partial rank correlations condition on the four genotype/virus
strata, with 49,999 within-stratum permutations and 2,000 paired stratified
bootstrap samples. All tests, missingness, sample sizes and leave-one-session-out
checks are exported. Actor-role tests have a single 432-test secondary family.

Coverage eligibility is fixed at 95%. Session 31101 has 94.4% coverage and is
excluded from primary inference, leaving 23 sessions. Its inclusion is reported
as descriptive sensitivity alongside episode-gap, bout-gap, movement/distance
and reused-stimulus-partner sensitivity checks.

Outputs go to `lgdel_usv_analysis/social_vocal_correlations/`: complete parameter
and statistics CSVs, provenance, source QA, validation, a French interpretation
report and all figures in PNG, PDF and SVG. `--figures-only` requires a valid
source-stamped numerical cache; `--recompute` explicitly rebuilds numeric results.
Figure styling changes do not invalidate numeric calculations.

Current interpretation: pooled anogenital correlations largely attenuate after
group adjustment, and none of the 312 primary tests reaches BH q<0.05. Two
secondary partner-escape measures associate with median call duration under BH,
but do not survive BY. All conclusions remain exploratory: behavior is scored
by geometric rules, only 23 sessions enter primary inference, and two stimulus
partners are reused. These between-session correlations do not test whether
calls are enriched during behavior episodes within a session.
