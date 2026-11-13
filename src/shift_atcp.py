"""Core methods for leakage-controlled source-to-target conformal experiments.

Important: importance-weighted conformal prediction has a finite-sample result
under known density ratios and covariate shift. Here ratios are estimated, so
coverage is approximate and must be described as such. Class-conditional
coverage is assessed with Mondrian target calibration, not inferred from a
marginal density-ratio argument.
"""
from __future__ import annotations
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold


def make_classifier(name: str, seed: int):
    if name == "logistic":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                 class_weight="balanced_subsample", n_jobs=-1, random_state=seed))
    if name == "hist_gradient_boosting":
        return make_pipeline(SimpleImputer(strategy="median"),
                             HistGradientBoostingClassifier(max_iter=250, learning_rate=.05,
                                 l2_regularization=1.0, random_state=seed))
    raise ValueError(f"Unknown model {name!r}")


class PlattCalibrator:
    """Platt scaling fitted on a partition disjoint from conformal calibration."""
    def __init__(self, seed=42):
        self.model = LogisticRegression(C=1e6, max_iter=2000, random_state=seed)

    def fit(self, p, y):
        p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
        z = np.log(p/(1-p)).reshape(-1, 1)
        self.model.fit(z, np.asarray(y, int))
        return self

    def predict(self, p):
        p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
        z = np.log(p/(1-p)).reshape(-1, 1)
        return self.model.predict_proba(z)[:, 1]


def positive_probability(model, X):
    p = model.predict_proba(X)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
    if 1 not in classes:
        raise ValueError("Classifier was not fitted with positive class 1")
    return p[:, classes.index(1)]


def conformal_quantile(scores, alpha=.10):
    """Split-conformal order statistic with the finite-sample +1 correction."""
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1-alpha)))
    if k > n:
        return float("inf")
    return float(np.partition(s, k-1)[k-1])


def weighted_conformal_quantile(scores, weights, test_weight, alpha=.10):
    """Weighted quantile with test-point mass; valid for known density ratios.

    Estimated ratios make the application approximate. Returns +inf if the
    required quantile lies beyond observed calibration mass.
    """
    s, w = np.asarray(scores, float), np.asarray(weights, float)
    if len(s) == 0 or len(s) != len(w):
        return float("inf")
    ok = np.isfinite(s) & np.isfinite(w) & (w > 0)
    s, w = s[ok], w[ok]
    if len(s) == 0:
        return float("inf")
    wt = max(float(test_weight), 1e-12)
    order = np.argsort(s)
    s, w = s[order], w[order]
    cutoff = (1-alpha) * (w.sum() + wt)
    if cutoff > w.sum():
        return float("inf")
    j = int(np.searchsorted(np.cumsum(w), cutoff, side="left"))
    return float(s[min(j, len(s)-1)])


def fit_density_ratio(source_X, target_adapt_X, target_test_X, seed=42,
                      clip=(.05, 20.0), n_splits=5):
    """Cross-fitted domain-classifier estimate of p_T(x)/p_S(x).

    Domain classifier is trained on common, harmonized predictors only; it
    does not add missingness indicators. OOF probabilities are used for source
    calibration weights. Test covariates are scored by the fold ensemble.
    Target test labels are never used.
    """
    Xs = np.asarray(source_X, float)
    Xa = np.asarray(target_adapt_X, float)
    Xtest = np.asarray(target_test_X, float)
    X = np.vstack([Xs, Xa])
    d = np.r_[np.zeros(len(Xs), int), np.ones(len(Xa), int)]
    if min(np.bincount(d)) < n_splits:
        raise ValueError("Too few source/adaptation rows for domain-classifier cross-fitting")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), float)
    test_fold_probs = []
    aucs = []
    for tr, va in cv.split(X, d):
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=seed))
        clf.fit(X[tr], d[tr])
        pv = clf.predict_proba(X[va])[:, 1]
        oof[va] = pv
        test_fold_probs.append(clf.predict_proba(Xtest)[:, 1])
    try:
        auc = float(roc_auc_score(d, oof))
    except ValueError:
        auc = float("nan")
    p_s = np.clip(oof[:len(Xs)], 1e-5, 1-1e-5)
    p_test = np.clip(np.mean(test_fold_probs, axis=0), 1e-5, 1-1e-5)
    # The domain classifier uses balanced class weights, yielding equal effective
    # source/target domain priors; its odds estimate p_T(x)/p_S(x) directly.
    ws = p_s/(1-p_s)
    wt = p_test/(1-p_test)
    lo, hi = clip
    ws_clip, wt_clip = np.clip(ws, lo, hi), np.clip(wt, lo, hi)
    ess = float(ws_clip.sum()**2 / np.sum(ws_clip**2)) if len(ws_clip) else 0.0
    diagnostics = {
        "domain_auc_oof": auc,
        "source_weight_ess": ess,
        "source_weight_ess_fraction": ess/max(len(ws_clip), 1),
        "source_weight_clip_fraction": float(np.mean(ws != ws_clip)),
        "test_weight_clip_fraction": float(np.mean(wt != wt_clip)),
        "source_weight_median": float(np.median(ws_clip)),
        "source_weight_p95": float(np.quantile(ws_clip, .95)),
    }
    return ws_clip, wt_clip, diagnostics



def wilson_lower(successes, n, confidence_z=1.645):
    """One-sided Wilson lower confidence bound for a binomial proportion."""
    if n <= 0: return 0.0
    phat = successes / n
    z2 = confidence_z**2
    den = 1 + z2/n
    center = phat + z2/(2*n)
    rad = confidence_z*np.sqrt((phat*(1-phat) + z2/(4*n))/n)
    return float(max(0.0, (center-rad)/den))


def prediction_sets(p, q0, q1):
    """Boolean membership matrix; q0/q1 can be scalars or per-row arrays."""
    p = np.asarray(p, float)
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    return np.column_stack([p <= q0, (1-p) <= q1])


def marginal_sets(p, q):
    """Standard binary split-conformal sets using one marginal threshold."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    return np.column_stack([p <= q, (1-p) <= q])


def evaluate_sets(y, sets, p, cost_fp=1., cost_fn=5., cost_referral=.5, sample_weight=None):
    """Coverage, efficiency, calibration and explicit set-based referral cost."""
    y, sets, p = np.asarray(y, int), np.asarray(sets, bool), np.asarray(p, float)
    w = np.ones(len(y), float) if sample_weight is None else np.asarray(sample_weight, float)
    if len(w) != len(y) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid evaluation weights")
    w = w / w.sum()
    sizes = sets.sum(axis=1)
    covered = sets[np.arange(len(y)), y]
    cls0, cls1 = y == 0, y == 1
    singleton0 = (sizes == 1) & sets[:, 0]
    singleton1 = (sizes == 1) & sets[:, 1]
    referral = sizes != 1
    fp = singleton1 & cls0
    fn = singleton0 & cls1
    row_cost = cost_fp*fp + cost_fn*fn + cost_referral*referral
    cov0 = np.average(covered[cls0], weights=w[cls0]) if cls0.any() else np.nan
    cov1 = np.average(covered[cls1], weights=w[cls1]) if cls1.any() else np.nan
    # Weighted expected calibration error using fixed-width probability bins.
    ece = 0.0
    for lo in np.linspace(0, 1, 11)[:-1]:
        hi = lo + .1
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if np.any(m):
            wm = w[m]; wm = wm / wm.sum()
            ece += float(w[m].sum()) * abs(float(np.average(y[m], weights=wm)) - float(np.average(p[m], weights=wm)))
    try: auc = float(roc_auc_score(y, p, sample_weight=w))
    except ValueError: auc = np.nan
    return {
        "n": int(len(y)), "prevalence": float(np.average(y, weights=w)),
        "marginal_coverage": float(np.average(covered, weights=w)),
        "class0_coverage": float(cov0), "diabetes_coverage": float(cov1),
        "class_coverage_gap": float(abs(cov0-cov1)) if np.isfinite(cov0) and np.isfinite(cov1) else np.nan,
        "mean_set_size": float(np.average(sizes, weights=w)),
        "singleton_rate": float(np.average(sizes == 1, weights=w)),
        "empty_set_rate": float(np.average(sizes == 0, weights=w)),
        "referral_rate": float(np.average(referral, weights=w)),
        "false_positive_singleton_rate": float(np.average(fp, weights=w)),
        "false_negative_singleton_rate": float(np.average(fn, weights=w)),
        "clinical_cost": float(np.average(row_cost, weights=w)),
        "brier": float(np.average((y-p)**2, weights=w)),
        "roc_auc": auc, "ece_10bin": float(ece),
    }


def bootstrap_cis(y, sets, p, n_boot=1000, seed=42, costs=(1.,5.,.5),
                  sample_weight=None, strata=None, psu=None):
    """Bootstrap CIs for all reported metrics.

    With PSU and strata, resample PSUs within each stratum (cluster bootstrap).
    Otherwise use participant-row bootstrap. For a complex survey, the PSU
    bootstrap is an approximation and should be checked against official
    replicate-weight methods where available.
    """
    rng = np.random.default_rng(seed)
    y, sets, p = np.asarray(y), np.asarray(sets), np.asarray(p)
    n = len(y)
    sw = np.ones(n, float) if sample_weight is None else np.asarray(sample_weight, float)
    keys = ["marginal_coverage", "class0_coverage", "diabetes_coverage", "class_coverage_gap",
            "mean_set_size", "referral_rate", "clinical_cost"]
    vals = {k: [] for k in keys}
    if (strata is None) != (psu is None):
        raise ValueError("Provide both strata and psu for design-aware bootstrap")
    if strata is not None:
        strata, psu = np.asarray(strata), np.asarray(psu)
        groups = {}
        for h in np.unique(strata):
            ix = np.where(strata == h)[0]
            clusters = np.unique(psu[ix])
            groups[h] = [ix[psu[ix] == c] for c in clusters]
    for _ in range(n_boot):
        if strata is None:
            idx = rng.integers(0, n, n)
        else:
            chosen = []
            for h, cluster_rows in groups.items():
                m = len(cluster_rows)
                if m <= 1:
                    chosen.extend(cluster_rows[0].tolist())
                    continue
                draw = rng.integers(0, m, m)
                for j in draw:
                    chosen.extend(cluster_rows[j].tolist())
            idx = np.asarray(chosen, int)
        met = evaluate_sets(y[idx], sets[idx], p[idx], *costs, sample_weight=sw[idx])
        for k in keys:
            vals[k].append(met[k])
    out = {}
    for k, v in vals.items():
        a = np.asarray(v, float); a = a[np.isfinite(a)]
        out[k+"_ci_low"] = float(np.quantile(a, .025)) if len(a) else np.nan
        out[k+"_ci_high"] = float(np.quantile(a, .975)) if len(a) else np.nan
    return out
