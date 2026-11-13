# Reviewer 1 comment-to-revision map

| Reviewer concern | Code / manuscript change | Evidence to report |
|---|---|---|
| SHIFT-ATCP collapses to target-only in external cohorts | Replace scalar trust heuristic with cross-fitted pointwise density-ratio weighted conformal baseline plus an independent target calibration gate. Gate falls back to target-only if weighted source fails a prespecified coverage/efficiency rule. | Gate pass rate, source-weight ESS, clipping fraction, complete per-seed results. Do not claim a benefit if gate always falls back. |
| Synthetic mixture only shows deterministic PSI weighting | Remove PSI-based scalar source weight. Estimate density ratios from a domain classifier. Add controlled covariate-shift and prevalence-shift experiments. | Known-vs-estimated ratio comparison; results across overlap and shift severity. |
| DP-SHIFT is heuristic class-conditional conformal | Include plain target-only Mondrian conformal as an explicit primary baseline. Avoid claiming class-conditional guarantees for marginal importance-weighted conformal. | Class-0 and diabetes-positive coverage, class gap, sample counts per class. |
| Best configuration selected on target test | No test-based model/feature/scenario selection. All combinations are exported; gate selection uses a separate target gate set. | `all_seed_results.csv`, split manifest, metadata. |
| CIs too narrow / not reproducible | Bootstrap test participants for marginal and class-specific coverage, class gap, set size, referral, and cost. Optional PSU-within-stratum bootstrap. | Per-run lower/upper CI columns and test n. |
| Tables 11, 12, 14, Fig. 6 inconsistent | Generate one row per seed × model × method × cost vector; generate summary only by model/method/cost. | Regenerate every table and figure from `all_seed_results.csv`; record data hashes and settings. |
| Kaggle provenance unclear | Replace primary source with official CDC BRFSS 2023. Keep Pima as secondary stress test. | Official source files, codebooks, download dates, SHA-256 hashes. |
| Shift metrics confounded by harmonization artefacts | Use one common feature set; harmonize age, BMI, hypertension, heart disease, smoking and outcome. Report missingness shift separately. | `shift_diagnostics.csv`, harmonization flow counts and variable definitions. |
| PSI and KL inconsistent | Compute PSI and both directed KL values on shared bins and verify PSI = KL(T||S)+KL(S||T). | `psi_identity_abs_error` should be numerically close to zero. |
| Outcome, hypertension, NHANES design unclear | Explicitly define self-reported diagnosed diabetes and variable mappings; retain BRFSS/NHANES survey design variables. | Harmonization script and codebook references. |
| Single seed/split | Run multiple prespecified seeds; preserve each seed's output. | Mean/SD across seeds plus per-seed participant/PSU bootstrap CIs. |
| Missing target-trained reference | Add target-trained classifier with independent target train, probability calibration, conformal calibration, and test partitions. | Target-trained marginal and Mondrian reference rows. |
| Missing density-ratio weighted conformal baseline | Add cross-fitted domain classifier and test-point-weighted quantile. | Importance-weighted source rows, domain AUC, ESS and clipping. |
| Missing sensitivity to clinical cost vector | Add three default cost vectors, configurable at runtime. | Results by cost vector; no single cost presented as universal. |
| Missing full external results grid | Retain all model/method/seed/cost combinations. | No best-test-only table. |
| Feature set C duplicates B | Use a single prespecified common predictor set for primary cross-survey experiment; any additional sets must be separate, genuinely distinct experiments. | Explicit feature list in metadata. |

## Non-negotiable interpretation rule

The revised implementation cannot guarantee that SHIFT-ATCP will outperform target-only conformal prediction. If the gate selects target-only in all real external runs, report that result plainly. A defensible contribution may then be the calibration-gated decision rule and a careful empirical characterization of when source data help, not a claim of universal improvement.
