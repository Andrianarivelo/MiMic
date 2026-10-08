# Directional vocal contexts: A acts on B versus B acts on A

Double-click `scripts/directional_social_calls.cmd`, run the Python script in an IDE, or use:

```powershell
python scripts/directional_social_calls.py
python scripts/directional_social_calls.py --figures-only
python scripts/directional_social_calls.py --recompute
```

Dependencies: NumPy, pandas, Matplotlib, statsmodels, threadpoolctl and the existing `social_call_probability.py` dependencies. A valid upstream numerical cache is required. Run that script first if its cache is missing or stale.

All editable numerical and styling parameters are at the beginning of the standalone generator. The figure covers anogenital sniffing, body sniffing and chasing. A means experimental resident, tracked mouse1; B means partner, tracked mouse2. WT/Het is A's genotype.

The top row shows paired boxstrips of probability of at least one dyad call onset within [-1,+1) s around behavior onset. Only recordings with complete windows in both actor directions contribute to each behavior. Narrow boxes show quartiles and median; every recording is shown, and paired lines connect its two directions. Hollow diamonds give the mean exact local-shift reference for the same selected recordings.

The bottom row estimates `(P A→B - local baseline A→B) - (P B→A - local baseline B→A)`. Positive effects favor the A-initiated context. The same cyclic call offsets are used in both directions, preserving the correlation of the outcomes and overlapping call windows. The two-sided tests compare the observed direction contrast with its joint shift distribution, centered on its exact expectation. Local stationarity within completely observed segments of 60 s blocks is assumed. Shift structure is not preserved across segment boundaries.

There are 19,999 shift combinations, 5,000 paired recording bootstraps within genotype, and a single nine-test exploratory family covering three behaviors and WT/Het/combined. BH, BY and Holm are exported. Combined estimates give equal weight to the two genotypes; each recording is weighted equally within genotype. At least five recordings and 20 complete episodes per direction are required; the combined analysis additionally requires two recordings per genotype. Unsupported subgroup estimates appear descriptive, without a plotted confidence interval or significance claim.

Outputs are saved to `lgdel_usv_analysis/directional_social_calls/`: PNG/PDF/SVG, per-recording direction measures, complete statistics, support/identity checks, source-stamped provenance and a French report. The numerical fingerprint is independent of style, allowing `--figures-only` reuse. Upstream data and raw metadata are never changed.

Actor direction can motivate hypotheses about caller roles but does not identify the vocal source. A-initiated context could contain A calls, B responses or both. Video-derived behavior labels and repeated stimulus partners remain methodological limitations.
