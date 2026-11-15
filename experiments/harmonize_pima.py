#!/usr/bin/env python3
"""Harmonize Pima Indians Diabetes to the SHIFT-ATCP v2 schema.

Pima is a female-only, high-prevalence cohort with only 3 of the 6
SHIFT-ATCP predictors measured (age, female, bmi). The other three
(hypertension, heart_disease, smoking_history) are absent and set to NaN.

Pima's 'Outcome' maps directly to 'diabetes' (1 = positive, 0 = negative).
BMI = 0 rows are treated as missing (NaN), because 0 is not a valid BMI.

Source: Kaggle Pima Indians Diabetes Database, originally from the
National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def harmonize_pima(path):
    d = pd.read_csv(path)
    required = ["Age", "BMI", "Outcome"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Pima file missing expected columns: {missing}")

    out = pd.DataFrame(index=d.index)
    out["diabetes"] = pd.to_numeric(d["Outcome"], errors="coerce").astype(float)
    out["age"] = pd.to_numeric(d["Age"], errors="coerce").astype(float)
    out["female"] = 1.0  # Pima cohort is female-only
    bmi = pd.to_numeric(d["BMI"], errors="coerce").astype(float)
    # BMI = 0 is physiologically impossible; treat as missing
    bmi = bmi.replace(0.0, np.nan)
    out["bmi"] = bmi
    # Not measured in Pima
    out["hypertension"] = np.nan
    out["heart_disease"] = np.nan
    out["smoking_history"] = np.nan
    # No survey design
    out["survey_weight"] = np.nan
    out["survey_strata"] = np.nan
    out["survey_psu"] = np.nan

    # Keep rows with valid outcome
    out = out.loc[out["diabetes"].notna()].copy()
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pima-csv", required=True)
    ap.add_argument("--outdir", default="data/processed")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t = harmonize_pima(a.pima_csv)
    t.to_csv(out / "pima_harmonized.csv", index=False)
    print("Pima", t.shape, "prevalence", round(float(t.diabetes.mean()), 5))
    print("Missing-rate per predictor:")
    print(
        t[["age", "female", "bmi", "hypertension", "heart_disease", "smoking_history"]]
        .isna()
        .mean()
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
