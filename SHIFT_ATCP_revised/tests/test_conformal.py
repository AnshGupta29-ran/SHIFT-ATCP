import numpy as np
from src.shift_atcp import (conformal_quantile, weighted_conformal_quantile,
    prediction_sets, marginal_sets, evaluate_sets, bootstrap_cis)


def test_split_quantile_finite_sample_plus_one():
    assert np.isinf(conformal_quantile(np.linspace(0,1,8), alpha=.1))
    assert 0 <= conformal_quantile(np.linspace(0,1,99), alpha=.1) <= 1


def test_weighted_quantile_includes_test_mass():
    assert np.isinf(weighted_conformal_quantile([.1,.2],[1,1],100,alpha=.1))
    assert weighted_conformal_quantile([.1,.2],[1,1],.01,alpha=.1) in (.1,.2)


def test_prediction_set_shapes_and_vector_thresholds():
    p=np.array([.1,.9])
    assert prediction_sets(p,.2,.2).tolist()==[[True,False],[False,True]]
    assert marginal_sets(p,np.array([.2,.2])).tolist()==[[True,False],[False,True]]


def test_metrics_include_cost_and_classwise_coverage():
    y=np.array([0,1,0,1]); p=np.array([.1,.9,.8,.2])
    sets=np.array([[1,0],[0,1],[0,1],[1,0]],dtype=bool)
    m=evaluate_sets(y,sets,p,cost_fp=1,cost_fn=5,cost_referral=.5)
    assert m['marginal_coverage']==.5
    assert m['diabetes_coverage']==.5
    assert m['clinical_cost'] > 0


def test_bootstrap_has_wide_plausible_ci_and_multiple_metrics():
    y=np.array([0]*90+[1]*10)
    p=np.r_[np.full(90,.1),np.full(10,.9)]
    sets=np.column_stack([p<=.2,(1-p)<=.2])
    ci=bootstrap_cis(y,sets,p,n_boot=100,seed=7)
    assert 'diabetes_coverage_ci_low' in ci
    assert ci['marginal_coverage_ci_high'] >= ci['marginal_coverage_ci_low']


def test_psi_equals_symmetric_kl_identity():
    from experiments.shift_diagnostics import psi_and_symmetric_kl
    rng=np.random.default_rng(3)
    s=rng.normal(0,1,500); t=rng.normal(.5,1.2,400)
    psi,kts,kst=psi_and_symmetric_kl(s,t)
    assert abs(psi-(kts+kst)) < 1e-10
