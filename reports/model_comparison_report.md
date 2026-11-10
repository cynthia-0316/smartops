# Model comparison and recommendation

*Issue #18. Numbers come from `results/model_results.csv`, `results/test_results.csv` and `notebooks/final_model_comparison.ipynb`.*

## Recommendation

Use a **tuned gradient-boosting model on the 6 raw sensors**. CatBoost scored highest (nested-CV ROC AUC 0.703), but the top models are practically tied (0.7018 to 0.7030), so the choice can be made on speed or convenience: LightGBM and HistGradientBoosting fit in a fraction of a second. Expect a **modest** prioritisation gain, not a way to skip most inspections: test-set AUC is about 0.71.

Do not add the 16 engineered columns (they did not help, #14). Choose the threshold from the cost of a missed fault versus an unneeded inspection:

| Threshold | Engines flagged | Precision | Recall | When to use |
|---|---|---|---|---|
| 0.50 | 76% | 0.69 | 0.84 | a missed fault costs about the same as an unneeded inspection |
| 0.65 | 46% | 0.80 | 0.58 | fewer inspections wanted, willing to miss more faults |

(Test set, CatBoost. Always flagging every engine gives precision 0.63 and recall 1.0.)

## Comparison on development data

Repeated stratified 5-fold CV (3 repeats, seed 42) on train + val (16,604 engines). Tuned rows use nested CV, so tuning does not inflate them. Full table: `reports/model_comparison_table.csv`; plot: `reports/figures/final_model_comparison.png`.

| Model | CV ROC AUC (mean +/- std) | Fit time (s) |
|---|---|---|
| CatBoost tuned | 0.7030 +/- 0.0085 | 28.0 (incl. search) |
| LightGBM tuned | 0.7025 +/- 0.0088 | 20.6 (incl. search) |
| GradientBoosting (sklearn) tuned | 0.7020 +/- 0.0088 | 181 (incl. search) |
| XGBoost tuned | 0.7018 +/- 0.0088 | 9.5 (incl. search) |
| HistGradientBoosting tuned | 0.7003 +/- 0.0092 | 7.4 (incl. search) |
| Default settings, best of five (GradientBoosting) | 0.7007 +/- 0.0085 | 1.3 |
| Default XGBoost | 0.6620 +/- 0.0109 | 0.08 |
| Dummy (no signal) | 0.5000 | 0.01 |

Tuning helps XGBoost a lot (0.662 to 0.702, its defaults overfit) and the others by 0.001 to 0.013.

### Which differences are real?

All tuned models sit within 0.003 AUC of each other, while the spread between folds is about 0.009. A corrected resampled t-test on the same 15 folds (CatBoost vs each other model) gives: LightGBM p = 0.18, GradientBoosting p = 0.038, XGBoost p = 0.012, HistGradientBoosting p = 0.010. After a Bonferroni correction for four comparisons (threshold 0.0125) only HistGradientBoosting and XGBoost differ from CatBoost, and only by 0.002 to 0.003 AUC. Statistically detectable in places, but too small to matter in practice. The fold AUCs in this test use parameters chosen on all development data, so they are a little optimistic; the p-values are only a guide to the differences, not to the absolute level.

## Test set (scored once)

The top 3 models by nested-CV AUC were refit on all development data and scored on the 2,931 held-out engines. Thresholds were fixed beforehand from development data (#15).

| Model | Test ROC AUC (95% bootstrap CI) | PR AUC |
|---|---|---|
| CatBoost tuned | 0.7085 (0.690 to 0.728) | 0.805 |
| LightGBM tuned | 0.7077 (0.690 to 0.726) | 0.805 |
| GradientBoosting tuned | 0.7044 (0.684 to 0.722) | 0.802 |

Test AUC is in line with the CV estimate (0.703), so there is no sign of overfitting to the development data. The three models cannot be told apart on 2,931 engines (the confidence intervals overlap almost completely).

## Limitations

- **Weak signal.** AUC is about 0.70. Inspecting the 10% of engines the model ranks highest is right 90% of the time, but random picking is already right 63% of the time (lift 1.4x). Under assumed costs the model beats "inspect everything" only when a missed fault costs about the same as an unneeded inspection (#15).
- **Unverified data provenance.** The data is a public Hugging Face copy. We do not know how it was collected or how "Needs Maintenance" is defined, so we cannot say whether the label reflects a real fault.
- **No time or engine ID.** Rows cannot be grouped by engine or ordered in time, so we could not run a grouped or forward-in-time split. If rows from the same engine appear in both training and test data, every score here may be optimistic.
- **Class balance.** 63% of engines are positive, so accuracy-style numbers look better than they are; compare against the "flag everyone" baseline (precision 0.63, recall 1.0).
- **Costs are assumptions.** The dataset has no inspection or failure costs; re-check the threshold with real figures.
- **Scope.** Issue #17 (logistic regression, random forest, ExtraTrees, kNN, SVM, MLP) was not complete when this was written, so non-boosting models are **not** compared here. The earlier baseline of about 0.688 AUC for logistic regression (PR #5) is lower than every tuned model above.
- **One test split, one scoring.** The test set was used exactly once, so there is no test-set tuning, but also only one estimate of test performance.
