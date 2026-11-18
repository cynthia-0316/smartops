# SmartOps decision dashboard

A static web dashboard (HTML, CSS, JavaScript) that ranks engines for inspection using the project's validated model. A planner can check one engine, upload a fleet CSV, see how sensors move the score, and compare cost trade-offs.

Workflow: **sensor readings or CSV -> model prediction -> interpretation -> threshold decision -> ranked inspection list**.

## Run it locally (no installs)

The exported model files are already in `dashboard/model/` (re-create them any time with `notebooks/dashboard_model_export.ipynb` in Google Colab, see below). Then either:

```bash
node dashboard/serve.js
```

and open http://localhost:8080, or just open `dashboard/index.html` in a browser. Node 18+ is only needed for `serve.js` and the tests; the page itself needs nothing.

## The model

**Spline logistic regression** (6 quantile knots per sensor, cubic B-splines, linear extrapolation, standardised, L2 logistic regression with `C = 0.03`), fitted on the **development data only** (train + val, 16,604 rows). The 2,931-row test set is scored once, for reporting.

It is an additive model: the six sensor effects add up exactly to the log-odds. That is why the dashboard can show an exact per-engine explanation without SHAP.

Why not the penalised GAM (20 splines) that the report names as its recommendation? On the test set the two are equivalent (AUC 0.7095 vs 0.7099, difference -0.0004 with 95% CI -0.002 to +0.001, see `experiments/RESULTS.md`). The spline logistic regression is a linear model on spline features, so it reproduces exactly in a browser from 48 coefficients plus the knots. A GAM would need the `pygam` library and a Python server. This is a deployment choice, not a claim that it is more accurate.

## How inference works (and how we know it is the real model)

1. `notebooks/dashboard_model_export.ipynb` (Colab) trains the model, scores the test set once, computes cross-validated metrics and out-of-fold decision tables, and exports the fitted knots, scaler and coefficients to `model/model.json` and `model/model_data.js`.
2. The notebook re-computes 256 reference probabilities from the exported numbers only and asserts they match scikit-learn (`< 1e-8`).
3. `js/core.js` re-implements the spline basis, scaling and logistic function. `tests/core.test.js` checks it against the same reference rows (`npm test` in this folder, or `node --test dashboard/tests/core.test.js`). The Model Insights page repeats this check in the browser.
4. All metrics shown (test AUC, cross-validation AUC, threshold and cost tables) are read from the exported file. Nothing is simulated or typed in by hand.

## Predict page: explanations and recommendations

After **Analyze Engine** the page shows five sections: A. maintenance assessment, B. why the engine was flagged, C. sensor contribution chart, D. recommended next steps, E. model limitations.

- Everything is derived from the fitted model's own per-sensor contributions (`Model.contributions()`), which add up exactly to the log-odds. A sensor is described as raising risk only when its contribution is at least +0.10 log-odds (odds multiplied by about 1.1 or more).
- Effects are shown as **odds multipliers** measured against each sensor's average effect in the development data, never as percentage-point changes. The model's *reference probability* (the logistic of the intercept, about 65%) is the prediction when every sensor effect is average; it is a modeling reference, **not** the observed share of labelled engines (about 63%) and not the risk of an actual average engine. If no sensor raises the score meaningfully, the text says so and points to that reference instead of inventing a reason.
- The single-engine page says "Inspection recommended — above selected threshold" rather than ranking priority. Priority ordering is done in Fleet Priorities, using risk score and inspection capacity.
- Recommendations are templates keyed to the sensors the model actually flagged. They are worded as areas to verify, never as a diagnosis or a repair claim. Model explanation, possible areas to investigate, recommended action and "confirmed diagnosis: not available" are kept separate.
- Readings outside the training range show a reliability warning, the prediction itself is unchanged, and the engine is never labeled "routine monitoring" on the strength of a low score alone.
- Fleet Priorities adds a **Main model driver** column and the CSV export adds `main_model_driver`, `driver_odds_multiplier`, `review_focus` and `reliability_note`.

## Business views

- The decision tables and the capture curve come from **out-of-fold predictions on development data**, never from the test set or in-sample fits.
- Costs are user assumptions (one unnecessary inspection = 1 unit, a missed case = r units). Dollar figures and savings are **hypothetical**.
- The dashboard says so when inspecting every engine is as cheap as the model (the dashboard computes where that happens from the saved tables).

## Deploy on Vercel (optional, not done)

The site is fully static, so no Python runtime is needed: inference runs in the browser and uploaded CSV files never leave it.

1. Import the repository in Vercel and set **Root Directory** to `dashboard`. No build command, output directory is the root.
2. `vercel.json` adds basic security headers.
3. Make sure `dashboard/model/model.json` and `model_data.js` are committed first.

Nothing has been deployed publicly. Deploying needs team approval.

## Tests

```bash
node --test dashboard/tests/core.test.js dashboard/tests/explain.test.js
```

Covers the spline maths (partition of unity, linear extrapolation), input validation, CSV parsing, ranking, export, threshold and cost logic. Fourteen of the 39 tests need the real `model/model.json`: parity with scikit-learn on 322 reference rows, contributions summing to the log-odds, saved metrics vs the project report (test AUC 0.7099, CV AUC about 0.703), and decision-table consistency. `tests/explain.test.js` covers the explanation and recommendation logic (see below). Without the model file the model-dependent tests are reported as skipped, not passed.

## Files

| Path | What |
|---|---|
| `index.html`, `css/styles.css` | Interface (Overview, Predict, Fleet Priorities, Model Insights, Business Impact, Responsible AI) |
| `js/core.js` | Model inference, validation, CSV, decision and cost maths (no dependencies) |
| `js/app.js` | User interface and charts |
| `model/` | Exported model (generated in Colab) |
| `tests/core.test.js` | Automated tests |
| `serve.js` | Tiny local web server |
| `vercel.json`, `package.json` | Deployment and convenience scripts |

## Limitations

- Provenance of the dataset and the label definition are unknown. Results do not show real-world performance.
- Test AUC is about 0.71, so the signal is modest. The dashboard is an inspection-prioritization aid with a human in the loop, not a safety system.
- Single predictions use the six raw sensors only. There is no engine ID or time information.
