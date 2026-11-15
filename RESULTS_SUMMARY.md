# Results summary — SHIFT-ATCP v2

Produced by the commands in `REPRODUCTION.md` on BRFSS 2023 → NHANES
2017–2018. Primary results are unweighted analytic-sample estimates.
Survey-weighted results are a sensitivity analysis.

## Dataset and shift

| Cohort | N | Diabetes prevalence |
|---|---:|---:|
| BRFSS 2023 (source) | 411,785 | 14.50% |
| NHANES 2017–2018 (target) | 5,393 | 16.24% |

Domain AUC (out-of-fold): **0.590** — moderate covariate shift.
Largest feature-level PSI: `age` = 0.086; `bmi` = 0.051; others < 0.01.
PSI = symmetric KL identity holds to ~1e-17.

## Table 1 — Primary experiment, hist_gradient_boosting

10 seeds; alpha = 0.10; bootstrap = 1000. Means over seeds.

| Method | Marginal cov | Diabetes cov | Class gap | Mean set size |
|---|---:|---:|---:|---:|
| source_only_marginal | 0.892 | 0.316 | 0.684 | 1.116 |
| **source_only_mondrian** | 0.895 | **0.874** | **0.024** | 1.372 |
| target_only_marginal | 0.907 | 0.409 | 0.591 | 1.163 |
| **target_only_mondrian** | 0.902 | **0.904** | **0.025** | 1.424 |
| importance_weighted_source_marginal | 0.882 | 0.253 | 0.747 | 1.088 |
| SHIFT_ATCP_calibration_gated | 0.896 | 0.340 | 0.660 | 1.128 |
| target_trained_marginal_reference | 0.899 | 0.360 | 0.640 | 1.180 |
| target_trained_mondrian_reference | 0.897 | 0.906 | 0.034 | 1.506 |

**Finding:** marginal conformal coverage sits near 0.90 for every method
while diabetes-positive coverage ranges from 0.25 to 0.41. Mondrian
conformal (source-only or target-only) recovers diabetes coverage to
~0.87–0.91 with class gap <= 0.04.

## Table 2 — Calibration gate behaviour

`gate_pass_rate` over 10 seeds (primary, unweighted):

| Model | gate_pass_rate |
|---|---:|
| hist_gradient_boosting | 0.5 |
| logistic | 0.6 |
| random_forest | 0.7 |

The gate falls back to target-only conformal in 30–50% of seeds.
SHIFT_ATCP does not outperform target-only Mondrian on diabetes-positive
coverage on the real cohorts. Per the README's scientific cautions, we
report this plainly.


## Table 2b — Pima stress test (reduced feature set)

Pima is a **female-only cohort** with only **3 of the 6** SHIFT-ATCP
predictors measured (`age`, `female`, `bmi`). Because `hypertension`,
`heart_disease`, and `smoking_history` are absent, the source model was
**retrained on the same 3 features** so the transport comparison is valid
within the reduced feature space. Pima results are **not directly comparable**
to the 6-feature BRFSS→NHANES analysis above.

10 seeds; α = 0.10; bootstrap = 1000; model = hist_gradient_boosting.
Domain AUC (OOF) = **0.934** — extreme shift.

| Method | Marginal cov | Diabetes cov | Class gap | Mean set size | Gate pass |
|---|---:|---:|---:|---:|---:|
| source_only_marginal | 0.673 | **0.040** | 0.960 | 1.022 | — |
| source_only_mondrian | 0.792 | **0.415** | 0.571 | 1.227 | — |
| target_only_marginal | 0.892 | 0.675 | 0.325 | 1.447 | — |
| **target_only_mondrian** | 0.905 | **0.907** | **0.060** | 1.536 | — |
| importance_weighted_source_marginal | 0.656 | **0.000** | 0.995 | 0.988 | — |
| SHIFT_ATCP_calibration_gated | 0.892 | 0.675 | 0.325 | 1.447 | **0.0** |
| target_trained_marginal_reference | 0.901 | 0.744 | 0.237 | 1.508 | — |
| target_trained_mondrian_reference | 0.900 | 0.894 | 0.068 | 1.588 | — |

**Findings:**

1. **Source-only marginal coverage collapses to 0.040 diabetes-positive
   coverage** under extreme shift (prevalence 14.5% → 34.9%). This is the
   most severe failure mode in the entire study.
2. **The calibration gate passes 0.0** — on every one of the 10 seeds, the
   gate correctly detected that source transport would be worse than
   target-only and fell back. The gate mechanism works as designed under
   extreme shift.
3. **`importance_weighted_source_marginal` produces 0.000 diabetes coverage** —
   the effective sample size collapsed to 12% of the source
   (`source_weight_ess_fraction = 0.123`), and the weighted quantile became
   useless for the minority class. This confirms the README's warning that
   the weighted-source method requires a valid covariate-shift assumption
   and known density ratios; under severe shift with estimated ratios, it
   fails.
4. **Target-only Mondrian still recovers class-conditional coverage** (0.907),
   same as for NHANES — but now at the cost of a larger set size (1.54 vs
   1.42) because the underlying shift is more severe.

**Feature-level shift:** PSI = 11.79 for `female` (Pima is 100% female) and
4.26 for `age` (Pima is younger-concentrated). These are structural
differences, not measurement differences: Pima is a non-representative
historical cohort, not a random sample of a modern population.

## Table 3 — Controlled-shift simulation

10 replicates per regime; source_n=5000; target_n=2000; bootstrap=200.

| Regime | Domain AUC | source_only_marginal (D-cov) | source_only_mondrian (D-cov) |
|---|---:|---:|---:|
| no_shift | 0.512 | 0.315 | **0.912** |
| covariate_moderate | 0.631 | 0.432 | **0.961** |
| covariate_severe | 0.784 | 0.637 | **0.984** |
| prevalence_shift | 0.512 | 0.267 | **0.886** |
| missingness_shift | 0.632 | 0.383 | **0.963** |

**Finding:** Mondrian conformal achieves class-conditional coverage
(0.89–0.98) under all regimes. Marginal coverage degrades under
prevalence shift, where the covariate-shift assumption is violated —
`importance_weighted_source_marginal` drops to 0.79 marginal coverage,
confirming the limitation stated in the methodology.

## Table 4 — Survey-weighted sensitivity

Test metrics shift modestly (marginal coverage 0.92–0.93 vs 0.89–0.91
unweighted) but the qualitative pattern is unchanged. Domain AUC
remains 0.590 because survey weights affect test metrics only; model
fitting and density-ratio estimation are unweighted.

## Verification

- `pytest tests/ -v`: **6 / 6 passed**, including the split-conformal
  quantile, weighted-quantile-with-test-mass, prediction-set, bootstrap,
  and PSI = KL identity tests.
- `run_metadata.json` records SHA-256 hashes of the harmonized inputs and
  all hyperparameters.
- Row counts: 720 rows in `results/brfss_to_nhanes/all_seed_results.csv`
  and in the survey-weighted variant; 1200 rows in
  `results/controlled_shift/all_simulation_results.csv`.
