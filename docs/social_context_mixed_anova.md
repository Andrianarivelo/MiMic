# Four-condition social figure: mixed ANOVA and Holm p values

Double-click `scripts/social_context_mixed_anova.cmd`, run in an IDE, or use:

```powershell
python scripts/social_context_mixed_anova.py
python scripts/social_context_mixed_anova.py --figures-only
python scripts/social_context_mixed_anova.py --recompute
```

The generator requires the validated `social_call_probability` numerical cache and the existing Python analysis dependencies (including SciPy and statsmodels). All routine parameters and styling settings are at the beginning of the script. Outputs use the independent directory `lgdel_usv_analysis/social_context_mixed_anova/`, preserving previous figures and raw data.

Only the partner-present window 300–900 s is analyzed. Both Inside and Outside belong to this window. As requested, all 12 WT and 12 Het recordings contribute; Het 31101, formerly excluded at the 95% tracking threshold, is included using only completely observed intervals. No missing bins are counted as Outside. The unchanged operational social definition is the union of 13 dyad behavior flags in fully observed 100 ms bins, with ≥50% occupancy required separately for each behavior flag. Probability is the fraction of bins with at least one call onset, frequency is calls per observed minute, and count is the unnormalized number of calls.

Each of three panels has WT Inside, WT Outside, Het Inside and Het Outside. Outside uses lighter blue/orange. Individual recordings appear as points with connecting paired lines, narrow quartile boxes and medians.

The statistical model is a two-way mixed ANOVA: genotype between recordings and context within recordings. The two-level split-plot decomposition tests genotype on recording means, and context plus interaction on Inside−Outside differences. Sum coding gives Type III tests; sphericity is automatic with two repeated levels. Every effect has F(1,22). The three ANOVA effects across all three metrics form a nine-test Holm family.

All six pairs of the four cells are tested for each metric. Within-genotype comparisons use paired t tests; between-genotype comparisons use Welch t tests. The 18 pairwise tests share one Holm correction across the figure. The labels `pH` mean Holm-adjusted p, never q. Raw p, adjusted p, effect sizes and pointwise recording-bootstrap CIs are exported. Virus is not a factor in the requested two-factor model.

Shapiro-Wilk residual checks and median-centered Levene checks are exported. The raw outcomes show strong skew and variance differences, so the classical parametric results are exploratory. HC3 covariance F tests provide a separate heteroscedasticity sensitivity. Correction for multiplicity does not fix these assumptions. Repeated stimulus partners may also induce dependence.

The script validates the ANOVA F statistics against an independent split-plot sums-of-squares calculation, paired sample counts and Holm bounds. Outputs include PNG/PDF/SVG, complete session metrics, ANOVA, all 18 comparisons, assumption diagnostics, HC3 sensitivity, source-partition checks, independent analysis checks, a report and provenance. Styling-only edits reuse the numerical cache with `--figures-only`.

## Simplified correction-comparison revision

The current renderer creates a simpler figure in `correction_comparison/<method>/`, preserving the original ANOVA-table figure. Only four simple comparisons are shown per panel; the two diagonal cell comparisons remain in the same complete 18-test family. Labels use `p=` and the small caption identifies the correction. ANOVA tables and assumption details remain in the CSVs and reports.

```powershell
python scripts/social_context_mixed_anova.py --figures-only --compare-corrections
python scripts/social_context_mixed_anova.py --figures-only --correction holm
python scripts/social_context_mixed_anova.py --figures-only --correction sidak
python scripts/social_context_mixed_anova.py --figures-only --correction bonferroni
python scripts/social_context_mixed_anova.py --figures-only --correction fdr_bh
```

The default is the exploratory Benjamini-Hochberg FDR version. It differs from familywise-error methods; it is not selected because it gives significance. All four methods use the same cached tests, recordings and correction family. Side-by-side statistical comparison CSVs and provenance are exported. The minimum adjusted p is 0.0590 for BH, 0.2759 for Sidak, and 0.3199 for Holm/Bonferroni, so none yields p<0.05. Sidak's independence assumption is not guaranteed for these related tests. The raw-scale distributional limitations remain applicable to every correction.
