#!/usr/bin/env python3
"""Controlled shift experiments using the same leakage-controlled pipeline.

Regimes: no shift, moderate/severe covariate shift (P(Y|X) fixed), prevalence
shift (intercept changes; covariate-shift assumption violated), and differential
missingness. This is a simulation benchmark, not clinical evidence.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.run_experiment import run_one, parse_cost_vectors


def make_cohort(n, seed, mean_shift=0., scale_shift=1., intercept=-2.0,
                missing_rate=0.):
    r=np.random.default_rng(seed)
    age_z=r.normal(mean_shift,scale_shift,n)
    bmi_z=r.normal(0,1,n)
    risk_z=r.normal(0,1,n)
    female=r.binomial(1,.52,n)
    logit=intercept + .55*age_z + .50*bmi_z + .25*risk_z + .10*female
    prob=1/(1+np.exp(-logit))
    y=r.binomial(1,prob)
    d=pd.DataFrame({"age":50+10*age_z,"female":female,"bmi":27+4*bmi_z,
                    "hypertension":r.binomial(1,1/(1+np.exp(-(.2+age_z*.35))),n),
                    "heart_disease":r.binomial(1,.10,n),
                    "smoking_history":r.binomial(1,.32,n),"diabetes":y})
    if missing_rate>0:
        mask=r.random(n)<missing_rate
        d.loc[mask,"bmi"]=np.nan
    return d


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--replicates',type=int,default=10)
    ap.add_argument('--source-n',type=int,default=5000)
    ap.add_argument('--target-n',type=int,default=2000)
    ap.add_argument('--bootstrap',type=int,default=200)
    ap.add_argument('--output',default='results/controlled_shift')
    a=ap.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    regimes={
        'no_shift':dict(mean_shift=0.,scale_shift=1.,intercept=-2.,missing_rate=0.),
        'covariate_moderate':dict(mean_shift=.5,scale_shift=1.,intercept=-2.,missing_rate=0.),
        'covariate_severe':dict(mean_shift=1.25,scale_shift=1.2,intercept=-2.,missing_rate=0.),
        'prevalence_shift':dict(mean_shift=0.,scale_shift=1.,intercept=-1.1,missing_rate=0.),
        'missingness_shift':dict(mean_shift=.5,scale_shift=1.,intercept=-2.,missing_rate=.25),
    }
    features=['age','female','bmi','hypertension','heart_disease','smoking_history']
    costs=parse_cost_vectors('default:1,5,0.5;balanced:1,2,0.5;high_referral:1,5,1')
    rows=[]
    for regime,kw in regimes.items():
        for rep in range(a.replicates):
            seed=7000+rep
            src=make_cohort(a.source_n,seed+100)
            tgt=make_cohort(a.target_n,seed+200,**kw)
            r=run_one(src,tgt,features,'diabetes','logistic',seed,.10,a.bootstrap,
                      costs,(.05,20.),.05,.02)
            for row in r:
                row['shift_regime']=regime
                row['replicate']=rep
                rows.append(row)
            print(f'completed {regime} replicate {rep+1}/{a.replicates}',flush=True)
    df=pd.DataFrame(rows); df.to_csv(out/'all_simulation_results.csv',index=False)
    summary=df.groupby(['shift_regime','method','cost_vector'],as_index=False).agg(
        replicates=('replicate','nunique'),marginal_coverage=('marginal_coverage','mean'),
        diabetes_coverage=('diabetes_coverage','mean'),class_coverage_gap=('class_coverage_gap','mean'),
        mean_set_size=('mean_set_size','mean'),referral_rate=('referral_rate','mean'),
        clinical_cost=('clinical_cost','mean'),gate_pass_rate=('gate_pass','mean'),
        domain_auc=('domain_auc_oof','mean'),source_weight_ess_fraction=('source_weight_ess_fraction','mean'))
    summary.to_csv(out/'simulation_summary.csv',index=False)
    print(summary.to_string(index=False))

if __name__=='__main__': main()
