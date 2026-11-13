#!/usr/bin/env python3
"""Feature-shift diagnostics on identical, harmonized predictor columns.

Reports source/target missingness separately from observed-value PSI and
symmetric KL. Continuous bins are defined from source quantiles. For a shared
binning scheme, PSI equals KL(target||source)+KL(source||target).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score


def _hist(x, edges):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    counts,_=np.histogram(x,bins=edges)
    p=counts.astype(float)+1e-8
    return p/p.sum()


def psi_and_symmetric_kl(source, target, bins=10):
    s=np.asarray(source,float); t=np.asarray(target,float)
    s=s[np.isfinite(s)]; t=t[np.isfinite(t)]
    if len(s)==0 or len(t)==0: return np.nan,np.nan,np.nan
    edges=np.unique(np.quantile(s,np.linspace(0,1,bins+1)))
    if len(edges)<3:
        vals=np.unique(np.r_[s,t])
        if len(vals)<=2: edges=np.array([-np.inf, .5, np.inf])
        else: edges=np.r_[-np.inf,(vals[:-1]+vals[1:])/2,np.inf]
    else:
        edges[0],edges[-1]=-np.inf,np.inf
    ps,pt=_hist(s,edges),_hist(t,edges)
    kl_ts=float(np.sum(pt*np.log(pt/ps)))
    kl_st=float(np.sum(ps*np.log(ps/pt)))
    psi=float(np.sum((pt-ps)*np.log(pt/ps)))
    return psi,kl_ts,kl_st


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--target',required=True)
    ap.add_argument('--features',required=True); ap.add_argument('--label',default='diabetes'); ap.add_argument('--out',default='results/shift_diagnostics.csv')
    a=ap.parse_args(); s=pd.read_csv(a.source); t=pd.read_csv(a.target)
    fs=[x.strip() for x in a.features.split(',') if x.strip()]
    rows=[{'feature':'__COHORT_SUMMARY__','source_n':len(s),'target_n':len(t),
           'source_prevalence':float(s[a.label].mean()),'target_prevalence':float(t[a.label].mean()),
           'prevalence_gap':float(t[a.label].mean()-s[a.label].mean()),
           'source_missing_rate':np.nan,'target_missing_rate':np.nan,'missingness_gap':np.nan,
           'psi':np.nan,'kl_target_source':np.nan,'kl_source_target':np.nan,
           'symmetric_kl':np.nan,'psi_identity_abs_error':np.nan,'domain_auc_oof':np.nan}]
    for f in fs:
        ps,ks,kt=psi_and_symmetric_kl(s[f],t[f])
        rows.append({'feature':f,'source_n':len(s),'target_n':len(t),'source_prevalence':np.nan,'target_prevalence':np.nan,'prevalence_gap':np.nan,'source_missing_rate':s[f].isna().mean(),'target_missing_rate':t[f].isna().mean(),
                     'missingness_gap':t[f].isna().mean()-s[f].isna().mean(),
                     'psi':ps,'kl_target_source':ks,'kl_source_target':kt,'symmetric_kl':ks+kt,
                     'psi_identity_abs_error':abs(ps-(ks+kt)),'domain_auc_oof':np.nan})
    # Cross-validated domain AUC on the common predictors only. Missingness is
    # measured above but is not encoded as a feature in this primary diagnostic.
    X=pd.concat([s[fs],t[fs]],ignore_index=True)
    d=np.r_[np.zeros(len(s),dtype=int),np.ones(len(t),dtype=int)]
    clf=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=3000))
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    oof=cross_val_predict(clf,X,d,cv=cv,method="predict_proba")[:,1]
    auc=float(roc_auc_score(d,oof))
    rows.append({'feature':'__DOMAIN_CLASSIFIER__','source_n':len(s),'target_n':len(t),'source_prevalence':np.nan,'target_prevalence':np.nan,'prevalence_gap':np.nan,'source_missing_rate':np.nan,'target_missing_rate':np.nan,
                 'missingness_gap':np.nan,'psi':np.nan,'kl_target_source':np.nan,
                 'kl_source_target':np.nan,'symmetric_kl':np.nan,'psi_identity_abs_error':np.nan,
                 'domain_auc_oof':auc})
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
