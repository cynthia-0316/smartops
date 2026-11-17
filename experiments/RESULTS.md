# Model improvement experiments (local)

Protocol: dev = train + val (16,604 rows), repeated stratified 5-fold x 3 (seed 42), ROC AUC. Per-fold
predictions saved by `harness.py` so models are compared fold-by-fold (corrected resampled t-test).
Scripts: `batch1.py` (model families), `batch2.py` (neural nets, boosting structure), `batch3.py`
(outliers, label-noise filtering, stacking), `nested_final.py` (honest nested-CV check of finalists).
All configs and scores: `log.csv`.

## What the data looks like
- The six sensors are almost independent of each other (largest |Spearman| 0.08), also within each class (0.09).
- Signal is additive: EBM with 15 pairwise interactions scores +0.0003 AUC over no interactions (p = 0.18);
  CatBoost depth 2 vs depth 1 +0.0008 (p = 0.46). Interaction importances are all at noise level.
- No leakage through row order (label autocorrelation equals chance).

## Nested-CV (honest) results
| Model | CV AUC |
|---|---|
| Stack (CatBoost + spline LR, both tuned inside folds) | 0.7029 +/- 0.0082 |
| Spline logistic regression (knots, C tuned inside folds) | 0.7026 +/- 0.0082 |
| CatBoost, 15-setting search inside folds | 0.7019 +/- 0.0084 |
| CatBoost nested (#13, larger search) | 0.7030 +/- 0.0085 |

## Test set (2,931 rows, scored once for these models; CatBoost row from #18)
| Model | Test AUC (95% CI) | diff vs CatBoost (95% CI) | PR AUC | Brier |
|---|---|---|---|---|
| Spline LR (6 knots, C = 0.03) | 0.7099 (0.692 to 0.729) | +0.0015 (-0.002 to +0.005) | 0.807 | 0.2044 |
| Stack CatBoost + spline LR | 0.7094 (0.691 to 0.728) | +0.0010 (-0.002 to +0.004) | 0.807 | 0.2049 |
| CatBoost tuned (#18) | 0.7085 (0.690 to 0.727) | 0 | 0.805 | 0.2046 |

## Tried, no gain
Linear LR (0.688), Gaussian NB (0.672; 0.701 after quantile transform), KDE naive Bayes (0.7025),
EBM (0.7028), kNN (0.697), random forest (0.699), extra trees (0.701), MLPs of several sizes (0.698 to 0.701),
bagged MLPs (0.701), CatBoost depth 1 to 4, LightGBM stumps / monotone constraints / linear trees (about 0.702),
5-seed CatBoost averaging (0.7037, optimistic params), winsorising outliers, label-noise filtering (worse),
simple and stacked ensembles (+0.0005, not significant). TabPFN not run: needs a Prior Labs account and licence.

## Round 2 (`batch4.py`, `batch5.py`, `torch_nets.py`)
- **No hidden structure:** decimal precision, RPM last digit, odd/even, fractional parts and RPM-as-category
  carry no extra label information (all p > 0.04 across 10+ checks; target-encoded RPM adds nothing).
- **Penalised GAM (pygam, smoothing chosen automatically inside each fit):** CV AUC 0.7037 to 0.7044 for
  8 to 30 splines; 20 splines 0.7041 +/- 0.0084. Best honest single model; vs nested stack +0.0012 (p = 0.018),
  vs spline LR +0.0009 (p = 0.21), vs tuned CatBoost +0.0005 (p = 0.58). Test AUC 0.7095 (0.692 to 0.728),
  identical to spline LR (diff -0.0004, CI -0.002 to +0.001).
- Per-feature knots, quadratic splines, one tensor interaction, monotone RPM: 0.7025 to 0.7044, no clear gain.
- **Modern neural nets** (piecewise-linear encoding + MLP, 5-net deep ensembles, early stopping): 0.7013 to 0.7032.
- **Ensembles** of GAM + CatBoost + neural net: 0.7045 to 0.7046 (+0.0005 over GAM, p > 0.3).

**Conclusion:** every family converges to CV AUC 0.703 to 0.705 and test AUC about 0.71; further
gains would likely need new information (more sensors, time series) rather than more complex models. Recommended: additive model (spline LR or penalised GAM).

## Reproduce
```bash
cd experiments
../.venv/bin/python batch1.py   # and batch2..5, torch_nets.py, nested_final.py (each appends to log.csv)
../.venv/bin/python collect_report_data.py && ../.venv/bin/python build_report.py   # rebuilds reports/smartops_results.html
```
Extra packages beyond requirements.txt: `interpret` (EBM), `pygam` (GAM), `torch` (neural nets).
