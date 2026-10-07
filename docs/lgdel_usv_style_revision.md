# Six-panel USV figure styling

Run `python scripts/restyle_lgdel_usv_figure.py`, open that file in an IDE, or
double-click `scripts/restyle_lgdel_usv_figure.cmd`.

The renderer preserves the supplied six-panel image's 0-300 s alone and
300-900 s partner windows. It uses a white background, thin axes, muted blue
and orange, thin horizontal stacked repertoire bars, and animal-level
boxplots with all individual observations. White diamonds indicate means;
box centers indicate medians and boxes span the interquartile range.
Timeline and vocal-output curves retain the original mean and SEM convention.

Outputs are isolated in `lgdel_usv_analysis/style_revision/`: PNG, PDF, SVG,
numeric CSVs, input hashes and provenance, and a before/after visual comparison.
The original outputs and current analysis tables are preserved. Configuration
is at the beginning of the script. `--figures-only` requires a valid numeric
cache; `--recompute` explicitly rebuilds it from original detector CSVs.

## Historical figure limitation

The adjacent manifest still has older inverted labels for 31097 and 31101.
The supplied figure and completed trial plan agree on 31097=WT and 31101=HET.
Reproduction applies those labels only in memory, without modifying metadata.
The 15-minute figure window differs from the separate matched-window analysis.

The displayed p values reproduce the original unadjusted, two-sided
animal-level Mann-Whitney tests. Repertoire p values reproduce the original
pooled-call chi-square annotations. Those composition tests do not account
for nesting of calls within animals, and some expected class counts are
small. They should be replaced by animal-level inference before a current
scientific interpretation. The styling workflow makes no new inference.
