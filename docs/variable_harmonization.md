# Primary BRFSS–NHANES harmonization dictionary

Outcome: self-reported clinician-diagnosed diabetes. This is not laboratory-defined total diabetes.

| Analysis variable | BRFSS 2023 source variable and mapping | NHANES 2017–2018 variable and mapping |
|---|---|---|
| `diabetes` | `DIABETE4`: 1→1; 3→0; 2 (pregnancy-only), 4 (pre-diabetes/borderline), 7, 9, missing excluded | `DIQ010`: 1→1; 2→0; borderline, 7, 9, missing excluded |
| `age` | `_AGE80`; retain age ≥20 | `RIDAGEYR`; retain age ≥20 |
| `female` | `_SEX`: 1 male→0; 2 female→1 | `RIAGENDR`: 1 male→0; 2 female→1 |
| `bmi` | `_BMI5` divided by 100 (two implied decimal places) | `BMXBMI` in kg/m² |
| `hypertension` | `BPHIGH6`: 1 yes→1; 3 no→0; pregnancy-only, borderline/pre-hypertension, DK/refused/missing treated as missing | `BPQ020`: 1 yes→1; 2 no→0; other values missing |
| `heart_disease` | `CVDINFR4` (heart attack) OR `CVDCRHD4` (angina/coronary heart disease): 1 if either yes; 0 only if both no; otherwise missing | `MCQ160D` (heart attack) OR `MCQ160C` (coronary heart disease): same composite rule |
| `smoking_history` | `SMOKE100`: 1 yes→1; 2 no→0; other values missing | `SMQ020`: 1 yes→1; 2 no→0; other values missing |
| Survey weight | `_LLCPWT` | `WTINT2YR` (interview weight, appropriate to self-reported questionnaire outcome) |
| Survey stratum | `_STSTR` | `SDMVSTRA` |
| PSU | `_PSU` | `SDMVPSU` |

## Important interpretation notes

- BRFSS `DIABETE4` and NHANES `DIQ010` capture self-reported diagnosed diabetes. They do not identify all undiagnosed diabetes.
- The hypertension variables are not identical instruments. The explicit harmonization above should be reported as an operational approximation and tested in sensitivity analyses.
- The heart-disease composite is harmonized as a broad history indicator; it is not a diagnosis-equivalent clinical endpoint.
- BMI missingness is retained and imputed within each training pipeline. Do not delete participants solely because BMI or another predictor is missing.
- The age restriction is applied to both cohorts to address age-eligibility differences.
- The BRFSS and NHANES survey designs differ. The current runner retains design variables and supports target test-weighted sensitivity metrics, but does not perform a fully design-weighted training/conformal procedure.
- Preserve the official raw XPT files and codebooks. Record the data release/version and file hashes in the study repository.
