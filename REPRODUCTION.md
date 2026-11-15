# Reproduction instructions — SHIFT-ATCP v2

Exact commands used to produce the results in `RESULTS_SUMMARY.md`.
Run from the repository root in a Python 3.11 environment with the
packages listed in `requirements.txt`.

## Environment

    conda create -n shift_atcp_v2 python=3.11 -y
    conda activate shift_atcp_v2
    pip install -r requirements.txt

## Data acquisition

Raw inputs are not tracked in git; download from the official sources.

- **BRFSS 2023**: `LLCP2023.XPT` from
  https://www.cdc.gov/brfss/annual_data/annual_2023.html
  (extracted from `LLCP2023XPT.zip`). The XPT inside the zip has a
  trailing space in its filename — rename to `LLCP2023.XPT`.
- **NHANES 2017–2018**: `DEMO_J.XPT`, `DIQ_J.XPT`, `BMX_J.XPT`,
  `BPQ_J.XPT`, `MCQ_J.XPT`, `SMQ_J.XPT` from
  https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017

Place all seven files in `data/raw/`.

## 1. Harmonize BRFSS and NHANES

    python -m experiments.harmonize_brfss_nhanes \
      --brfss-xpt data/raw/LLCP2023.XPT \
      --nhanes-demo data/raw/DEMO_J.XPT \
      --nhanes-diq data/raw/DIQ_J.XPT \
      --nhanes-bmx data/raw/BMX_J.XPT \
      --nhanes-bpq data/raw/BPQ_J.XPT \
      --nhanes-mcq data/raw/MCQ_J.XPT \
      --nhanes-smq data/raw/SMQ_J.XPT \
      --outdir data/processed

Expected output:
- BRFSS: 411,785 rows, prevalence 0.14499
- NHANES: 5,393 rows, prevalence 0.16243

## 2. Shift diagnostics

    python -m experiments.shift_diagnostics \
      --source data/processed/brfss_2023_harmonized.csv \
      --target data/processed/nhanes_2017_2018_harmonized.csv \
      --features age,female,bmi,hypertension,heart_disease,smoking_history \
      --out results/brfss_nhanes/shift_diagnostics.csv

Key numbers: domain AUC (OOF) = 0.596; largest PSI on `age` (0.086).
PSI = symmetric KL identity verified to ~1e-17.

## 3. Main experiment (primary, unweighted)

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

Runtime: ~12 hours on CPU. Output: 720 result rows
(10 seeds × 3 models × 8 methods × 3 cost vectors).

## 4. Survey-weighted sensitivity analysis

    python -m experiments.run_experiment \
      --source data/processed/brfss_2023_harmonized.csv \
      --target data/processed/nhanes_2017_2018_harmonized.csv \
      --label diabetes \
      --features age,female,bmi,hypertension,heart_disease,smoking_history \
      --models logistic,random_forest,hist_gradient_boosting \
      --seeds 42,43,44,45,46,47,48,49,50,51 \
      --alpha 0.10 --bootstrap 1000 \
      --target-split 0.15,0.15,0.20,0.10,0.20,0.20 \
      --target-weight survey_weight --target-strata survey_strata --target-psu survey_psu \
      --output results/brfss_to_nhanes_surveyweighted

Survey weights affect test metrics and bootstrap resampling only; they do
not change model fitting, density-ratio estimation, or conformal calibration.

## 5. Controlled-shift simulation

    python -m experiments.simulate_shift_regimes \
      --replicates 10 --source-n 5000 --target-n 2000 --bootstrap 200 \
      --output results/controlled_shift

Five regimes: no_shift, covariate_moderate, covariate_severe,
prevalence_shift, missingness_shift.

## 6. Test suite

    pytest tests/ -v

6 tests pass: split-conformal quantile, weighted quantile with test mass,
prediction set shapes, metric definitions, bootstrap CIs, PSI = KL identity.

## Observations on the gate

Across the 10-seed primary run, the calibration gate (`gate_pass_rate`)
was not 1.0:

| Model | gate_pass_rate |
|---|---|
| hist_gradient_boosting | 0.5 |
| logistic | 0.6 |
| random_forest | 0.7 |

So the gate falls back to target-only conformal for 30–50% of seeds.
Per the README's "Scientific cautions", source transport is not claimed
to improve the real external results.

## Reproducibility

- `run_metadata.json` records SHA-256 hashes of both harmonized CSVs, the
  feature list, seeds, split fractions, `weight_clip`, `gate_tolerance`,
  and `min_efficiency_gain`.
- `split_manifest.csv` records exact row indices assigned to each
  partition for every seed.

## 7. Pima stress test (reduced feature set)

Pima is a **female-only cohort** with only **3 of the 6** SHIFT-ATCP
predictors measured (`age`, `female`, `bmi`). `hypertension`,
`heart_disease`, and `smoking_history` are absent. To keep the transport
comparison valid, the source model is **retrained on the same 3 features**
rather than filled with sentinel values. Pima results are reported as a
separate stress test and are not pooled with the BRFSS→NHANES primary
analysis.

Raw file: `data/raw/diabetes.csv` (768 rows, 9 columns: Pregnancies,
Glucose, BloodPressure, SkinThickness, Insulin, BMI,
DiabetesPedigreeFunction, Age, Outcome).

### Harmonize

    python -m experiments.harmonize_pima \
      --pima-csv data/raw/diabetes.csv \
      --outdir data/processed

Expected: 768 rows, prevalence 0.349, BMI missing 1.4% (zeros → NaN),
hypertension / heart_disease / smoking_history / survey_* all 100% NaN.

### Shift diagnostics

    python -m experiments.shift_diagnostics \
      --source data/processed/brfss_2023_harmonized.csv \
      --target data/processed/pima_harmonized.csv \
      --features age,female,bmi \
      --out results/brfss_pima/shift_diagnostics.csv

Key numbers: domain AUC (OOF) = 0.934. PSI = 11.79 for `female`
(Pima is 100% female) and 4.26 for `age`. These reflect structural
composition differences, not measurement differences.

### Main Pima run

    python -m experiments.run_experiment \
      --source data/processed/brfss_2023_harmonized.csv \
      --target data/processed/pima_harmonized.csv \
      --label diabetes \
      --features age,female,bmi \
      --models hist_gradient_boosting \
      --seeds 42,43,44,45,46,47,48,49,50,51 \
      --alpha 0.10 --bootstrap 1000 \
      --target-split 0.15,0.15,0.20,0.10,0.20,0.20 \
      --output results/brfss_to_pima

Runtime: ~2–3 hours on CPU. Output: 720 result rows
(10 seeds × 1 model × 8 methods × 3 cost vectors).

Note on the calibration gate: `gate_pass_rate = 0.0` for every method
under this extreme shift. The gate correctly fell back to target-only
conformal in all 10 seeds. Report this plainly — it is the intended
behaviour, not a failure.
