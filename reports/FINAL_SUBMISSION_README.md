# SmartOps final analysis: deliverables and reproduction

## Which file is which

| File | Role |
|---|---|
| `reports/SmartOps_Final_Colab_Analysis.html` | **Final submission version (merged).** Results first (the SmartOps Model Lineup page: overview, risk curves, 9 families, test set, lessons, SmartOps use, 99-run table), then the Code Appendix: the **executed Colab notebook** (22 code cells with their real outputs, 19 tables, 8 charts, 16/16 checks) rendered in the same style, followed by sections A to M with all repo code. Built by `tools/build_model_lineup_with_code.js` and `tools/render_notebook.js`. |
| `reports/colab_run/SmartOps_Final_Colab_Analysis.html` | **Strict nbconvert export** of the same executed notebook (Jupyter layout with the SmartOps palette), kept as the plain "HTML from the executed notebook" record. Produced by the `Export Colab HTML` GitHub Actions workflow (`tools/export_colab_html.py`). |
| `reports/colab_run/SmartOps_Final_Colab_Analysis.executed.ipynb` | The notebook exactly as executed in Google Colab (outputs included). Source of both HTML files. |
| `notebooks/SmartOps_Final_Colab_Analysis.ipynb` | The same notebook without outputs (what Colab opens from GitHub). |

How the HTML was made: the notebook was **executed in Google Colab** (second full run, Python 3.13, scikit-learn 1.6.1). The HTML files were then **converted outside Colab** from that executed `.ipynb` (nbconvert on a GitHub Actions runner, and a Node renderer for the merged page). No cell was rerun and no output was edited; the export script refuses to run if the code cells differ from the committed notebook, if any output is an error, or if secret-like text is present. Colab-only scripts attached to DataFrame displays (they call `google.colab`, break outside Colab and only add an "interactive table" button) are removed; the tables themselves are untouched.

If the instructor insists on a file downloaded from inside a Colab session, run the last cell of the notebook in Colab; it performs the same nbconvert step there.

## Run in Google Colab (no local install)

1. Open <https://colab.research.google.com/github/cynthia-0316/smartops/blob/feat/final-colab-analysis/notebooks/SmartOps_Final_Colab_Analysis.ipynb>
2. **Runtime > Run all** (about 15 to 25 minutes on the free CPU runtime). The notebook installs `pygam` and `catboost`, downloads the data from Hugging Face and clones this repository for the saved experiment artifacts.
3. When every cell has finished, the last cell exports the executed notebook with `jupyter nbconvert` and downloads `SmartOps_Final_Colab_Analysis.html` and the executed `.ipynb`.
4. If the download is blocked: **File > Save a copy in Drive**, then run `!jupyter nbconvert --to html "<path to the copy>"` after mounting Drive.
5. Check the final cell of Section M: it prints PASS/FAIL for each verification check. Any FAIL is reported, not hidden.

`REPO_REF` in the second code cell selects the branch the artifacts are read from. After this branch is merged, change it to `main`.

## What is rerun and what is loaded

| Item | Status |
|---|---|
| Dataset shapes, class balance, missing values, duplicates, outliers, EDA | Fresh |
| Splits, leakage checks, 15 CV folds | Fresh |
| Baselines and 7 representative models (LR, spline LR, Gaussian NB, kNN, RF, ET, HistGB, CatBoost with the #13 settings) | Fresh, compared with the saved runs (same folds and seeds) |
| Nested-CV spline LR | Fresh |
| Test-set evaluation of spline LR, GAM, CatBoost; bootstrap intervals; ROC; confusion matrix | Fresh (test file opened once) |
| Permutation importance, response curves | Fresh (train/validation only) |
| Thresholds and hypothetical cost scenarios | Fresh (development out-of-fold only) |
| 99 configurations (all families) | **Historical**, loaded from `experiments/log.csv` and `results/model_results.csv`; not rerun |
| Boosting finalists' test predictions | **Historical**, AUC recomputed from the saved predictions; models not refit |
| EBM, pyGAM sweeps, PyTorch nets, stacks, XGBoost/LightGBM tuning | **Historical** (hours of compute); source scripts are in the code appendix |

## Verification summary

Done locally (Windows machine with no Python; nothing was installed):

- The lineup page loads without console errors; 13 appendix sections, 38 code blocks, all highlighted on expand; 99 rows in the runs table; no horizontal overflow in the desktop or mobile viewport.
- Every appendix block is generated from the committed files, so nothing is abbreviated or hand-typed.
- The page keeps PR #28's removal of the 0.707 "ceiling" claim (searched: no occurrence).
- Numbers shown in the new text come from the repository: test AUC 0.7099 (spline LR) and 0.7095 (GAM), diff -0.0004 (CI -0.002 to +0.001), CV 0.7032 / nested 0.7026 / GAM 0.7041 (`experiments/RESULTS.md`, `reports/smartops_models.html`).
- Wording change: the base page called the GAM the "recommended model". It now says the GAM has the best single CV score and that the dashboard deploys the spline logistic regression for transparency and lightweight browser deployment, with no superiority claim.
- No secrets or tokens in any added file.

### Colab run (first execution, 2026-11-18)
The notebook ran top to bottom in Colab (Python 3.13, scikit-learn 1.6.1, pandas 2.2.3, pygam 0.12.0, catboost 1.2.10) with no cell errors, and all 16 checks in Section M printed PASS:

- 19,535 = 16,604 development + 2,931 test; no duplicates, no missing values; train/val/test share no feature rows; test file opened once.
- Fresh 15-fold CV agrees with the saved runs to within 0.0002 (spline LR 0.7032, nested spline LR 0.7026, CatBoost #13 settings 0.7037, logistic regression 0.6876, Gaussian NB 0.7009, kNN 0.6970, RF 0.6988, ET 0.7010).
- Fresh test AUC: spline LR 0.7099 (CI 0.692 to 0.729), GAM 0.7095, CatBoost #13 settings 0.7082 (saved: 0.7085). Paired bootstrap spline LR minus GAM +0.0004 (CI -0.0014 to +0.0023): no superiority claim.
- Stored test predictions of the six finalists recompute to the saved AUCs.
- Cost scenarios (hypothetical): at a miss cost 3x or more the cost-optimal rule flags ~everyone and saves nothing versus inspecting all.

**Conversion checks (GitHub Actions, then local browser):** all 16 PASS lines present; no traceback; 8 charts embedded and loaded; code visible; no Colab API calls left; no secret-like strings; merged page renders 22 code cells, 19 tables, 8 images, no console errors, no horizontal overflow, results above the code.
