#!/usr/bin/env python3
"""Create harmonized BRFSS 2023 and NHANES 2017-2018 analytic CSVs.

BRFSS input: CDC 2023 combined landline/cell SAS transport .XPT file.
NHANES input: six 2017-2018 component XPT files: DEMO_J, DIQ_J, BMX_J,
BPQ_J, MCQ_J, SMQ_J. Use the official CDC/NCHS files and preserve the raw
files and codebooks for provenance. Output outcome is self-reported
diagnosed diabetes, not laboratory-defined total diabetes.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def read_xpt(path):
    return pd.read_sas(path, format="xport", encoding="latin1")


def binary_yes_no(series, yes=1, no=2):
    x = pd.to_numeric(series, errors="coerce")
    return x.map({yes:1, no:0}).astype(float)


def composite_any_yes(a, b):
    a = pd.to_numeric(a, errors="coerce"); b = pd.to_numeric(b, errors="coerce")
    out = pd.Series(np.nan, index=a.index, dtype=float)
    out[(a == 1) | (b == 1)] = 1.0
    out[(a == 2) & (b == 2)] = 0.0
    return out


def harmonize_brfss(path):
    d = read_xpt(path)
    required = ["DIABETE4","_AGE80","_SEX","_BMI5","BPHIGH6","CVDINFR4","CVDCRHD4",
                "SMOKE100","_LLCPWT","_STSTR","_PSU"]
    missing = [c for c in required if c not in d.columns]
    if missing: raise ValueError(f"BRFSS file missing expected 2023 variables: {missing}")
    out = pd.DataFrame(index=d.index)
    age = pd.to_numeric(d["_AGE80"], errors="coerce")
    diabetes = pd.to_numeric(d["DIABETE4"], errors="coerce").map({1:1,3:0})
    out["diabetes"] = diabetes
    out["age"] = age
    sex = pd.to_numeric(d["_SEX"], errors="coerce")
    out["female"] = sex.map({1:0,2:1})
    # CDC _BMI5 has two implied decimal places in the raw SAS transport file.
    out["bmi"] = pd.to_numeric(d["_BMI5"], errors="coerce") / 100.0
    out["hypertension"] = pd.to_numeric(d["BPHIGH6"], errors="coerce").map({1:1,3:0})
    out["heart_disease"] = composite_any_yes(d["CVDINFR4"], d["CVDCRHD4"])
    out["smoking_history"] = binary_yes_no(d["SMOKE100"],1,2)
    out["survey_weight"] = pd.to_numeric(d["_LLCPWT"], errors="coerce")
    out["survey_strata"] = d["_STSTR"].astype(str)
    out["survey_psu"] = d["_PSU"].astype(str)
    # Align adult eligibility with NHANES 2017-2018 adult interview measures.
    out = out.loc[age >= 20].copy()
    out = out.loc[out["diabetes"].notna() & (out["survey_weight"] > 0)].copy()
    return out.reset_index(drop=True)


def harmonize_nhanes(demo, diq, bmx, bpq, mcq, smq):
    frames = [read_xpt(x) for x in (demo,diq,bmx,bpq,mcq,smq)]
    merged = frames[0]
    for f in frames[1:]:
        if "SEQN" not in f.columns: raise ValueError("NHANES component file lacks SEQN")
        merged = merged.merge(f, on="SEQN", how="left", validate="one_to_one")
    req = ["RIDAGEYR","RIAGENDR","WTINT2YR","SDMVSTRA","SDMVPSU","DIQ010",
           "BMXBMI","BPQ020","MCQ160C","MCQ160D","SMQ020"]
    missing = [c for c in req if c not in merged.columns]
    if missing: raise ValueError(f"Merged NHANES files missing variables: {missing}")
    out = pd.DataFrame(index=merged.index)
    age = pd.to_numeric(merged["RIDAGEYR"], errors="coerce")
    out["age"] = age
    out["female"] = pd.to_numeric(merged["RIAGENDR"], errors="coerce").map({1:0,2:1})
    out["bmi"] = pd.to_numeric(merged["BMXBMI"], errors="coerce")
    out["diabetes"] = pd.to_numeric(merged["DIQ010"], errors="coerce").map({1:1,2:0})
    out["hypertension"] = binary_yes_no(merged["BPQ020"],1,2)
    out["heart_disease"] = composite_any_yes(merged["MCQ160D"], merged["MCQ160C"])
    out["smoking_history"] = binary_yes_no(merged["SMQ020"],1,2)
    # DIQ010 borderline, don't-know, refused and missing are excluded by map.
    out["survey_weight"] = pd.to_numeric(merged["WTINT2YR"], errors="coerce")
    out["survey_strata"] = merged["SDMVSTRA"].astype(str)
    out["survey_psu"] = merged["SDMVPSU"].astype(str)
    out = out.loc[age >= 20].copy()
    out = out.loc[out["diabetes"].notna() & (out["survey_weight"] > 0)].copy()
    return out.reset_index(drop=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--brfss-xpt", required=True)
    ap.add_argument("--nhanes-demo", required=True); ap.add_argument("--nhanes-diq", required=True)
    ap.add_argument("--nhanes-bmx", required=True); ap.add_argument("--nhanes-bpq", required=True)
    ap.add_argument("--nhanes-mcq", required=True); ap.add_argument("--nhanes-smq", required=True)
    ap.add_argument("--outdir", default="data/processed")
    a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    s=harmonize_brfss(a.brfss_xpt)
    t=harmonize_nhanes(a.nhanes_demo,a.nhanes_diq,a.nhanes_bmx,a.nhanes_bpq,a.nhanes_mcq,a.nhanes_smq)
    s.to_csv(out/"brfss_2023_harmonized.csv",index=False)
    t.to_csv(out/"nhanes_2017_2018_harmonized.csv",index=False)
    print("BRFSS",s.shape,"prevalence",s.diabetes.mean())
    print("NHANES",t.shape,"prevalence",t.diabetes.mean())
    print("Predictors: age,female,bmi,hypertension,heart_disease,smoking_history")
    print("Outcome: self-reported diagnosed diabetes (not laboratory-defined total diabetes)")

if __name__=="__main__": main()
