# WT versus Het: calls inside and outside social behavior

Run `scripts/social_inside_outside_figure.cmd` by double-clicking, or launch the Python script in an IDE. Analysis and figure settings are at its beginning.

```powershell
python scripts/social_inside_outside_figure.py
python scripts/social_inside_outside_figure.py --figures-only
python scripts/social_inside_outside_figure.py --recompute
```

This standalone generator reads the validated `social_call_probability` trace cache. If that cache is absent or stale, run `scripts/social_call_probability.py` first. Dependencies are NumPy, pandas, Matplotlib, statsmodels and the existing upstream analysis dependencies.

The white six-panel figure compares WT (blue) and Het (orange), with one point per recording, thin quartile boxes and median lines. The top row shows calls inside any of the 13 social behavior flags; the bottom row uses the complementary observed intervals, while the partner is still present. Each upstream behavior flag requires at least 50% occupancy of a completely observed 100 ms bin. This is an operational bin classification, not frame-exact assignment or caller localization.

The three columns show probability of at least one call onset per 100 ms, calls per observed minute, and unnormalized call totals. All three use the same exhaustive inside/outside partition. The script verifies conservation of calls and observed bins for every recording. Eligibility follows the upstream 95% coverage threshold, giving 12 WT and 11 Het recordings. Each category requires at least five observed seconds per recording.

Six two-sided genotype-label permutation tests preserve group sizes within virus, with a common BH family. BY and Holm, Monte Carlo uncertainty, effect sizes and genotype/virus-stratified recording bootstrap intervals are also exported. Genotype exchangeability conditional on virus and independent recordings are assumptions. Reused stimulus partners can create residual dependence. The brackets compare genotypes within a state; they do not test inside against outside.

Outputs are in `lgdel_usv_analysis/social_inside_outside/`: PNG/PDF/SVG figure, per-recording metrics, six-test statistics, conservation checks, results report and fingerprinted provenance. Numeric settings and upstream input hashes govern cache reuse. Styling-only edits work with `--figures-only` without numerical recomputation. Raw data, metadata and upstream results are not changed.
