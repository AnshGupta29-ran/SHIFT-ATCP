# Changelog

## v2 – Reviewer-driven methodological revision

- Replaced scalar PSI-derived source-trust weighting with cross-fitted pointwise density-ratio estimation.
- Added test-point mass to the weighted conformal quantile.
- Added an independent labelled target gate; the gate selects weighted-source transport only when a prespecified Wilson coverage bound and set-size improvement criterion are both met. Otherwise it falls back to target-only conformal.
- Separated probability-calibration fitting, conformal-score calibration, gate selection, and final testing.
- Randomly allocates target partitions without using target labels to form the partitions.
- Added source-only and target-only marginal/Mondrian baselines, importance-weighted source marginal baseline, and target-trained marginal/Mondrian references.
- Added full seed × model × method × cost-vector output; removed best-on-test selection.
- Added bootstrap intervals for marginal/class-specific coverage, class gap, set size, referral rate, and clinical cost; optional PSU-within-stratum cluster bootstrap.
- Added multiple clinical cost vectors and Brier/AUROC/ECE metrics.
- Added BRFSS 2023–NHANES 2017–2018 harmonization script, with common predictor definitions and retained survey design variables.
- Added PSI/KL identity checks, missingness diagnostics, out-of-fold domain AUC, and controlled shift simulation.
- Added Overleaf-ready revised methodology and reviewer concern-to-code mapping.

All previous manuscript numerical results are superseded and must be regenerated.
