#!/usr/bin/env python3
"""Leakage-controlled SHIFT-ATCP experiments.

Target partitions are assigned by random row allocation, not outcome
stratification, so the covariate-only adaptation partition does not use target
labels. Target labels are used only in explicitly named target-supervised
partitions: gate, target-model training, target probability calibration, and
conformal calibration. The final test labels are never used for fitting,
calibration, gating, or model selection.
"""
from __future__ import annotations
import argparse, json, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.shift_atcp import (make_classifier, positive_probability, PlattCalibrator,
    conformal_quantile, weighted_conformal_quantile, fit_density_ratio,
    prediction_sets, marginal_sets, evaluate_sets, bootstrap_cis, wilson_lower)


def split_source(df, label, seed):
    """Source splits: 60% train, 10% probability calibration, 30% conformal."""
    work = df.copy(); work["_source_row_id"] = np.arange(len(work))
    a, rest = train_test_split(work, train_size=.60, stratify=work[label], random_state=seed)
    b, c = train_test_split(rest, train_size=.25, stratify=rest[label], random_state=seed+1)
    return [x.reset_index(drop=True) for x in (a,b,c)]


def split_target_unstratified(df, seed, fractions=(.15,.15,.20,.10,.20,.20)):
    """Target split; no target labels are consulted for partition assignment."""
    names = ["adapt", "gate", "train", "calfit", "conf", "test"]
    if len(fractions) != len(names) or not np.isclose(sum(fractions),1.0):
        raise ValueError("target split must contain six fractions summing to 1")
    work = df.copy(); work["_target_row_id"] = np.arange(len(work))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(work))
    n = len(work)
    cuts = [int(round(sum(fractions[:i+1])*n)) for i in range(len(fractions)-1)] + [n]
    out, start = {}, 0
    for name, stop in zip(names, cuts):
        out[name] = work.iloc[order[start:stop]].reset_index(drop=True)
        start = stop
    return out


def make_split_manifest(source, target, label, seeds, target_fractions):
    rows=[]
    for seed in seeds:
        s_train,s_calfit,s_conf=split_source(source,label,seed)
        for part,df in zip(("train","calfit","conf"),(s_train,s_calfit,s_conf)):
            rows.extend({"seed":seed,"cohort":"source","partition":part,"row_index":int(i)} for i in df["_source_row_id"])
        t=split_target_unstratified(target,seed+10000,target_fractions)
        for part,df in t.items():
            rows.extend({"seed":seed,"cohort":"target","partition":part,"row_index":int(i)} for i in df["_target_row_id"])
    return pd.DataFrame(rows)


def _fit_model_and_calibrator(train, calfit, features, label, model_name, seed):
    model = make_classifier(model_name, seed)
    model.fit(train[features], train[label].astype(int))
    cal = PlattCalibrator(seed).fit(positive_probability(model, calfit[features]), calfit[label])
    return model, cal


def _class_thresholds(p, y, alpha):
    y = np.asarray(y, int); p = np.asarray(p, float)
    return (conformal_quantile(p[y == 0], alpha),
            conformal_quantile(1-p[y == 1], alpha))


def _gate_source_transport(y_gate, p_gate, q_weighted_gate, q_target, alpha, tolerance, min_efficiency_gain):
    """Pre-test gate: use weighted source only if gate coverage is acceptable
    and it improves set size over target-only marginal CP by a fixed margin.
    Gate and conformal calibration partitions are disjoint.
    """
    sw_sets = marginal_sets(p_gate, q_weighted_gate)
    tgt_sets = marginal_sets(p_gate, q_target)
    sw = evaluate_sets(y_gate, sw_sets, p_gate)
    tg = evaluate_sets(y_gate, tgt_sets, p_gate)
    # Use an unweighted one-sided 90% Wilson lower bound for the gate; target
    # survey weights are reserved for final test-set metric estimation.
    y_gate = np.asarray(y_gate, int)
    covered = sw_sets[np.arange(len(y_gate)), y_gate]
    lower = wilson_lower(int(covered.sum()), len(y_gate), confidence_z=1.645)
    passed = (lower >= 1-alpha-tolerance and
              sw["mean_set_size"] <= tg["mean_set_size"]-min_efficiency_gain)
    sw["gate_coverage_wilson90_lower"] = lower
    return bool(passed), sw, tg


def run_one(source, target, features, label, model_name, seed, alpha, n_boot,
            cost_vectors, weight_clip, gate_tolerance, min_efficiency_gain,
            target_weight=None, target_strata=None, target_psu=None,
            target_fractions=(.15,.15,.20,.10,.20,.20)):
    s_train, s_calfit, s_conf = split_source(source, label, seed)
    t = split_target_unstratified(target, seed+10000,target_fractions)
    # Source-trained model and independent source probability-calibration fit.
    smodel, scal = _fit_model_and_calibrator(s_train, s_calfit, features, label, model_name, seed)
    ps_conf = scal.predict(positive_probability(smodel, s_conf[features]))
    ps_gate = scal.predict(positive_probability(smodel, t["gate"][features]))
    ps_test = scal.predict(positive_probability(smodel, t["test"][features]))
    ys_conf = s_conf[label].to_numpy(int)
    yt_conf = t["conf"][label].to_numpy(int)
    yt_gate = t["gate"][label].to_numpy(int)
    yt_test = t["test"][label].to_numpy(int)
    pt_conf = scal.predict(positive_probability(smodel, t["conf"][features]))

    # Source marginal and Mondrian thresholds.
    src_scores = 1 - np.where(ys_conf == 1, ps_conf, 1-ps_conf)
    q_source_marg = conformal_quantile(src_scores, alpha)
    q_source_0, q_source_1 = _class_thresholds(ps_conf, ys_conf, alpha)

    # Target conformal calibration is independent of gate and model fitting.
    q_target_marg = conformal_quantile(1-np.where(yt_conf == 1, pt_conf, 1-pt_conf), alpha)
    q_target_0, q_target_1 = _class_thresholds(pt_conf, yt_conf, alpha)

    # Estimated density ratio uses source conformal X and target adaptation X.
    # Target test X is only queried to obtain its individual ratio; labels unused.
    ws, wt, shift_diag = fit_density_ratio(s_conf[features], t["adapt"][features],
                                           t["test"][features], seed+77, clip=weight_clip)
    q_weighted_test = np.array([weighted_conformal_quantile(src_scores, ws, wi, alpha) for wi in wt])
    # For the independent gate, estimate test-point ratios from the same fitted
    # cross-fit domain ensemble by scoring gate covariates in a second call.
    _, wg, _ = fit_density_ratio(s_conf[features], t["adapt"][features],
                                 t["gate"][features], seed+77, clip=weight_clip)
    q_weighted_gate = np.array([weighted_conformal_quantile(src_scores, ws, wi, alpha) for wi in wg])
    gate_pass, gate_weighted_metrics, gate_target_metrics = _gate_source_transport(
        yt_gate, ps_gate, q_weighted_gate, q_target_marg, alpha,
        gate_tolerance, min_efficiency_gain)

    # Target-trained reference: separate target train/calibration-fit partitions.
    tmodel, tcal = _fit_model_and_calibrator(t["train"], t["calfit"], features,
                                             label, model_name, seed+900)
    pt_conf_model = tcal.predict(positive_probability(tmodel, t["conf"][features]))
    pt_test_model = tcal.predict(positive_probability(tmodel, t["test"][features]))
    qt0, qt1 = _class_thresholds(pt_conf_model, yt_conf, alpha)
    qt_marg_model = conformal_quantile(1-np.where(yt_conf == 1, pt_conf_model, 1-pt_conf_model), alpha)

    # All methods are frozen before final test labels are accessed.
    methods = {
        "source_only_marginal": ("marginal", q_source_marg, ps_test),
        "target_only_marginal": ("marginal", q_target_marg, ps_test),
        "importance_weighted_source_marginal": ("marginal_vector", q_weighted_test, ps_test),
        "SHIFT_ATCP_calibration_gated": ("marginal_vector", q_weighted_test if gate_pass else np.full(len(ps_test), q_target_marg), ps_test),
        "source_only_mondrian": ("mondrian", (q_source_0,q_source_1), ps_test),
        "target_only_mondrian": ("mondrian", (q_target_0,q_target_1), ps_test),
        "target_trained_mondrian_reference": ("mondrian_target_model", (qt0,qt1), pt_test_model),
        "target_trained_marginal_reference": ("marginal_target_model", qt_marg_model, pt_test_model),
    }
    rows = []
    tw = t["test"][target_weight].to_numpy(float) if target_weight else None
    ts = t["test"][target_strata].to_numpy() if target_strata else None
    tp = t["test"][target_psu].to_numpy() if target_psu else None
    for method, (kind, q, ptest) in methods.items():
        if kind == "marginal": sets = marginal_sets(ptest, q)
        elif kind == "marginal_vector": sets = marginal_sets(ptest, q)
        elif kind == "mondrian": sets = prediction_sets(ptest, q[0], q[1])
        elif kind == "mondrian_target_model": sets = prediction_sets(ptest, q[0], q[1])
        else: sets = marginal_sets(ptest, q)
        for cost_name, (cfp,cfn,cref) in cost_vectors.items():
            metrics = evaluate_sets(yt_test, sets, ptest, cfp,cfn,cref,tw)
            ci = bootstrap_cis(yt_test, sets, ptest, n_boot, seed+31,
                               (cfp,cfn,cref), tw, ts, tp)
            rows.append({"seed":seed,"model":model_name,"method":method,
                         "cost_vector":cost_name,"alpha":alpha,"gate_pass":gate_pass,
                         "gate_weighted_coverage":gate_weighted_metrics["marginal_coverage"],
                         "gate_weighted_set_size":gate_weighted_metrics["mean_set_size"],
                         "gate_coverage_wilson90_lower":gate_weighted_metrics["gate_coverage_wilson90_lower"],
                         "gate_target_coverage":gate_target_metrics["marginal_coverage"],
                         "source_n":len(source),"target_n":len(target),
                         "target_adapt_n":len(t["adapt"]),"target_gate_n":len(t["gate"]),
                         "target_train_n":len(t["train"]),"target_calfit_n":len(t["calfit"]),
                         "target_conf_n":len(t["conf"]),"target_test_n":len(t["test"]),
                         "source_conf_class0_n":int(np.sum(ys_conf==0)),
                         "source_conf_class1_n":int(np.sum(ys_conf==1)),
                         "target_gate_class0_n":int(np.sum(yt_gate==0)),
                         "target_gate_class1_n":int(np.sum(yt_gate==1)),
                         "target_conf_class0_n":int(np.sum(yt_conf==0)),
                         "target_conf_class1_n":int(np.sum(yt_conf==1)),
                         "target_test_class0_n":int(np.sum(yt_test==0)),
                         "target_test_class1_n":int(np.sum(yt_test==1)),
                         **shift_diag,**metrics,**ci})
    return rows


def parse_cost_vectors(s):
    # Format: default:1,5,0.5;balanced:1,2,0.5;high_referral:1,5,1
    out = {}
    for item in s.split(";"):
        name, vals = item.split(":",1)
        nums = tuple(float(x) for x in vals.split(","))
        if len(nums) != 3 or min(nums) < 0: raise ValueError("Each cost vector needs nonnegative FP,FN,referral costs")
        out[name] = nums
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--target", required=True)
    ap.add_argument("--label", default="diabetes")
    ap.add_argument("--features", required=True)
    ap.add_argument("--models", default="logistic,random_forest,hist_gradient_boosting")
    ap.add_argument("--seeds", default=",".join(map(str,range(42,52))))
    ap.add_argument("--alpha", type=float, default=.10)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--weight-clip", default="0.05,20")
    ap.add_argument("--gate-tolerance", type=float, default=.05)
    ap.add_argument("--min-efficiency-gain", type=float, default=.02)
    ap.add_argument("--cost-vectors", default="default:1,5,0.5;balanced:1,2,0.5;high_referral:1,5,1")
    ap.add_argument("--target-weight", default=None)
    ap.add_argument("--target-strata", default=None)
    ap.add_argument("--target-psu", default=None)
    ap.add_argument("--target-split", default="0.15,0.15,0.20,0.10,0.20,0.20",
                    help="Fractions: adapt,gate,target-train,target-calfit,target-conf,test")
    ap.add_argument("--output", default="results/experiment")
    args = ap.parse_args()
    features = [v.strip() for v in args.features.split(",") if v.strip()]
    clip = tuple(float(x) for x in args.weight_clip.split(","))
    if len(clip)!=2 or clip[0] <= 0 or clip[1] < clip[0]: raise ValueError("Invalid weight clipping bounds")
    if not 0 < args.alpha < 1: raise ValueError("alpha must be in (0,1)")
    if bool(args.target_weight) != bool(args.target_strata) or bool(args.target_weight) != bool(args.target_psu):
        raise ValueError("For survey-aware evaluation, supply target-weight, target-strata, and target-psu together")
    source, target = pd.read_csv(args.source), pd.read_csv(args.target)
    required = features + [args.label]
    for nm, df in (("source",source),("target",target)):
        miss = [c for c in required if c not in df.columns]
        if miss: raise ValueError(f"{nm} missing columns {miss}")
        df[args.label] = pd.to_numeric(df[args.label],errors="raise").astype(int)
        if set(df[args.label].dropna().unique()) != {0,1}:
            raise ValueError(f"{nm} outcome must contain both classes coded 0 and 1")
        for c in features: df[c] = pd.to_numeric(df[c],errors="coerce")
        bad = [c for c in features if df[c].notna().sum()==0]
        if bad: raise ValueError(f"{nm} all-missing predictors: {bad}")
    if target[args.label].isna().any() or source[args.label].isna().any(): raise ValueError("Outcome contains missing values")
    if args.target_weight:
        for c in [args.target_weight,args.target_strata,args.target_psu]:
            if c not in target: raise ValueError(f"Target survey-design column {c!r} missing")
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    costs = parse_cost_vectors(args.cost_vectors)
    target_fractions=tuple(float(x) for x in args.target_split.split(","))
    if len(target_fractions)!=6 or min(target_fractions)<=0 or not np.isclose(sum(target_fractions),1.0):
        raise ValueError("--target-split needs six positive fractions summing to one")
    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    make_split_manifest(source,target,args.label,seeds,target_fractions).to_csv(out/"split_manifest.csv",index=False)
    allrows=[]
    for model in [x.strip() for x in args.models.split(",") if x.strip()]:
        for seed in seeds:
            print(f"model={model} seed={seed}",flush=True)
            allrows.extend(run_one(source,target,features,args.label,model,seed,args.alpha,args.bootstrap,
                costs,clip,args.gate_tolerance,args.min_efficiency_gain,
                args.target_weight,args.target_strata,args.target_psu,target_fractions))
    res = pd.DataFrame(allrows)
    res.to_csv(out/"all_seed_results.csv",index=False)
    group = res.groupby(["model","method","cost_vector"],as_index=False).agg(
        runs=("seed","nunique"), marginal_coverage_mean=("marginal_coverage","mean"),
        marginal_coverage_sd=("marginal_coverage","std"),
        diabetes_coverage_mean=("diabetes_coverage","mean"),
        diabetes_coverage_sd=("diabetes_coverage","std"),
        class_coverage_gap_mean=("class_coverage_gap","mean"),
        mean_set_size=("mean_set_size","mean"),referral_rate=("referral_rate","mean"),
        clinical_cost_mean=("clinical_cost","mean"),brier_mean=("brier","mean"),
        roc_auc_mean=("roc_auc","mean"),ece_10bin_mean=("ece_10bin","mean"),
        domain_auc_oof_mean=("domain_auc_oof","mean"),
        source_weight_ess_fraction_mean=("source_weight_ess_fraction","mean"),
        gate_pass_rate=("gate_pass","mean"))
    group.to_csv(out/"summary_by_model_method_cost.csv",index=False)
    # One row per model/seed/method/cost; no best-on-test selection or aggregation
    # across different models, features, or shifts is performed.
    metadata = {"source_file":str(args.source),"target_file":str(args.target),
        "source_sha256":hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
        "target_sha256":hashlib.sha256(Path(args.target).read_bytes()).hexdigest(),
        "features":features,"label":args.label,"models":args.models,"seeds":args.seeds,"target_split_fractions":target_fractions,
        "alpha":args.alpha,"bootstrap":args.bootstrap,"weight_clip":clip,
        "gate_tolerance":args.gate_tolerance,"min_efficiency_gain":args.min_efficiency_gain,
        "cost_vectors":costs,"python":platform.python_version(),"source_rows":len(source),
        "target_rows":len(target),"source_prevalence":float(source[args.label].mean()),
        "target_prevalence":float(target[args.label].mean()),
        "source_predictor_missingness":{f:float(source[f].isna().mean()) for f in features},
        "target_predictor_missingness":{f:float(target[f].isna().mean()) for f in features},
        "target_survey_design":bool(args.target_weight),
        "selection_rule":"SHIFT-ATCP gate is determined on a labelled target gate partition; final test labels are never used.",
        "coverage_note":"Known-ratio weighted conformal has a covariate-shift guarantee; estimated density ratios make source-weighted results approximate. Target-only split conformal/Mondrian is the validity baseline."}
    (out/"run_metadata.json").write_text(json.dumps(metadata,indent=2,default=str))
    print(group.to_string(index=False))

if __name__=="__main__": main()
