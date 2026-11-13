# SHIFT-ATCP revised research implementation (v2)

This package implements a leakage-controlled revision of the original diabetes conformal-prediction experiments, explicitly responding to Reviewer 1's technical concerns in the IEEE Access decision letter.

## What changed

1. **No scalar PSI-based source-trust heuristic.** Source transport uses a cross-fitted domain classifier to estimate pointwise density ratios. Test-point mass is included in the weighted conformal quantile. Ratios are estimated, so the weighted method is described as approximate, not as an exact finite-sample guarantee.
2. **Independent data partitions.** Probability-calibration fitting and conformal-score calibration use disjoint observations. Target rows are allocated randomly without using target labels for partition assignment. A separate labelled target gate chooses weighted-source transport or falls back to target-only calibration. Final test labels are never used for selection.
3. **Explicit baselines.** Source-only marginal and Mondrian, target-only marginal and Mondrian, importance-weighted source marginal, calibration-gated SHIFT-ATCP, and target-trained marginal/Mondrian reference models are all reported.
4. **Full results grid.** Results are retained by seed, model, method, and cost vector. There is no best-on-target-test selection and no averaging across unrelated configurations.
5. **Correct uncertainty reporting.** Bootstrap intervals are produced for marginal and class-specific coverage, class gap, set size, referral rate, and clinical cost. Optional PSU-within-stratum cluster bootstrap is available for the target test set.
6. **Clinical-cost sensitivity.** False-positive, false-negative, and referral costs are varied using prespecified vectors.
7. **Shift diagnostics.** PSI, both directed KL terms, missingness differences, and out-of-fold domain AUC are reported on identical common predictors. The PSI/KL identity is checked numerically.
8. **Cohort provenance and harmonization.** The primary pipeline uses official BRFSS 2023 and NHANES 2017–2018 data with explicit self-reported diagnosed-diabetes definitions. Pima is secondary only; the undocumented Kaggle cohort is not the primary source.

## Scientific cautions

- The outcome is **self-reported diagnosed diabetes**, not laboratory-defined total diabetes. Undiagnosed cases are not captured.
- The weighted-source method has a formal weighted-conformal interpretation only with known density ratios and a valid covariate-shift assumption. Here density ratios are estimated; report this as approximate.
- Marginal importance weighting does not guarantee diabetes-positive class-conditional coverage. The main class-conditional baseline is ordinary target-only Mondrian conformal prediction.
- The gate uses labelled target observations. This is a limited-label adaptation protocol, not a zero-label domain adaptation method.
- If the gate always falls back to target-only conformal in the real cohorts, report that plainly. Do not claim that source transport improved the real external results.
- The primary model/conformal analysis is an analytic-sample analysis. Optional target survey weights affect final test metrics only; they are not used in model fitting or density-ratio estimation. Therefore weighted metrics are a sensitivity analysis, not a complete survey-design conformal guarantee.
- Pima is a small, female-only cohort with a different feature set. It must be reported as a separate stress test, not pooled into the BRFSS–NHANES primary table.

## Installation

Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Data acquisition

Download the **2023 BRFSS SAS Transport data file and codebook** from the [official CDC 2023 BRFSS page](https://www.cdc.gov/brfss/annual_data/annual_2023.html). Download the NHANES 2017–2018 XPT component files from the official NCHS pages: DEMO_J, DIQ_J, BMX_J, BPQ_J, MCQ_J, and SMQ_J. Keep the raw files and codebooks unchanged in `data/raw/` and record their hashes.

The BRFSS harmonizer uses `_AGE80`, `_SEX`, `_BMI5`, `DIABETE4`, `BPHIGH6`, `CVDINFR4`, `CVDCRHD4`, `SMOKE100`, `_LLCPWT`, `_STSTR`, and `_PSU`. The NHANES harmonizer uses `RIDAGEYR`, `RIAGENDR`, `BMXBMI`, `DIQ010`, `BPQ020`, `MCQ160C`, `MCQ160D`, `SMQ020`, `WTINT2YR`, `SDMVSTRA`, and `SDMVPSU`.

## Harmonize BRFSS and NHANES

```bash
python -m experiments.harmonize_brfss_nhanes \
  --brfss-xpt data/raw/LLCP2023.XPT \
  --nhanes-demo data/raw/DEMO_J.XPT \
  --nhanes-diq data/raw/DIQ_J.XPT \
  --nhanes-bmx data/raw/BMX_J.XPT \
  --nhanes-bpq data/raw/BPQ_J.XPT \
  --nhanes-mcq data/raw/MCQ_J.XPT \
  --nhanes-smq data/raw/SMQ_J.XPT \
  --outdir data/processed
```

The script aligns both cohorts to age >=20 and creates the same six predictors: `age,female,bmi,hypertension,heart_disease,smoking_history`. Missing predictor values are retained for imputation inside training folds. The outcome maps diagnosed diabetes to 1 and no diabetes to 0; pregnancy-only, borderline, unknown, refused, and missing responses are excluded.

## Shift diagnostics

```bash
python -m experiments.shift_diagnostics \
  --source data/processed/brfss_2023_harmonized.csv \
  --target data/processed/nhanes_2017_2018_harmonized.csv \
  --features age,female,bmi,hypertension,heart_disease,smoking_history \
  --out results/brfss_nhanes/shift_diagnostics.csv
```

## Main experiment

```bash
python -m experiments.run_experiment \
  --source data/processed/brfss_2023_harmonized.csv \
  --target data/processed/nhanes_2017_2018_harmonized.csv \
  --label diabetes \
  --features age,female,bmi,hypertension,heart_disease,smoking_history \
  --models logistic,random_forest,hist_gradient_boosting \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --alpha 0.10 --bootstrap 1000 \
  --target-split 0.15,0.15,0.20,0.10,0.20,0.20 \
  --output results/brfss_to_nhanes
```

For a **survey-weighted test-metric sensitivity analysis**, additionally pass:

```bash
  --target-weight survey_weight \
  --target-strata survey_strata \
  --target-psu survey_psu
```

This changes test metric estimates and uses PSU-within-stratum bootstrap resampling. It does not change model fitting, domain-ratio estimation, or conformal calibration. Report unweighted analytic-sample results as primary unless the complete survey design is handled consistently.

## Controlled shift experiments

```bash
python -m experiments.simulate_shift_regimes \
  --replicates 10 --source-n 5000 --target-n 2000 --bootstrap 200 \
  --output results/controlled_shift
```

This compares no shift, moderate/severe covariate shift (with invariant `P(Y|X)`), prevalence shift (which violates the covariate-shift assumption), and missingness shift. Simulation results are methodological stress tests, not clinical evidence.

## Main output files

- `all_seed_results.csv`: complete row-level experiment grid (seed × model × method × cost vector).
- `split_manifest.csv`: exact source/target row indices assigned to each partition for every seed.
- `summary_by_model_method_cost.csv`: descriptive mean/SD across seeds, retaining model/method/cost distinctions.
- `run_metadata.json`: data-file SHA-256 hashes, features, splits, seeds, settings, and guarantee qualifications.
- `shift_diagnostics.csv`: feature-level PSI/KL, missingness, and domain AUC.

## Revised methodology

See [`methodology.tex`](methodology.tex) for an Overleaf-ready draft and [`docs/reviewer_comment_implementation.md`](docs/reviewer_comment_implementation.md) for a reviewer concern-to-code map. All manuscript tables and figures must be regenerated from the revised outputs. Do not reuse any numerical result from the previous manuscript without rerunning and tracing it to the new output files.
