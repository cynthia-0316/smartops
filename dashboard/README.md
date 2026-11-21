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

**Penalised logistic GAM** (pyGAM `LogisticGAM`, one cubic P-spline term with 20 splines per sensor, smoothing chosen by a lambda grid, linear extrapolation outside the training range), fitted on the **development data only** (train + val, 16,604 rows). The 2,931-row test set is scored once, for reporting. It is the same GAM as in `notebooks/SmartOps_Final_Colab_Analysis.ipynb` (CV AUC 0.7041, test AUC 0.7095).

It is an additive model: the six sensor effects add up exactly to the log-odds. That is why the dashboard can show an exact per-engine explanation without SHAP.

A GAM stores, per sensor, a knot vector and 20 coefficients, so the browser can reproduce it exactly from 120 coefficients plus the knots; no Python server is needed. The spline logistic regression used in an earlier version of this dashboard scores the same on the test set (AUC 0.7099 vs 0.7095, difference not significant), so the choice between them is about consistency with the final report, not accuracy.

## How inference works (and how we know it is the real model)

1. `notebooks/dashboard_model_export_gam.ipynb` (Colab) trains the GAM, scores the test set once, computes cross-validated metrics and out-of-fold decision tables, and exports the knots, coefficients and basis means to `model/model.json` and `model/model_data.js`.
2. The notebook re-computes 394 reference probabilities (250 real test rows plus 144 edge cases, 26 of them outside the training range) from the exported numbers only and asserts they match pyGAM (`< 1e-8`) before it runs the slow steps.
3. `js/core.js` re-implements the spline basis and logistic function. `tests/core.test.js` checks it against the same reference rows (`npm test` in this folder, or `node --test dashboard/tests/core.test.js`). The Model Insights page repeats this check in the browser.
4. All metrics shown (test AUC, cross-validation AUC, threshold and cost tables) are read from the exported file. Nothing is simulated or typed in by hand.

## Predict page: explanations and recommendations

After **Analyze Engine** the page shows five sections: A. maintenance assessment, B. why the engine was flagged, C. sensor contribution chart, D. recommended next steps, E. model limitations.

- Everything is derived from the fitted model's own per-sensor contributions (`Model.contributions()`), which add up exactly to the log-odds. A sensor is described as raising risk only when its contribution is at least +0.10 log-odds (odds multiplied by about 1.1 or more).
- Effects are shown as **odds multipliers** measured against each sensor's average effect in the development data, never as percentage-point changes. The model's *reference probability* (the logistic of the intercept, about 65%) is the prediction when every sensor effect is average; it is a modeling reference, **not** the observed share of labelled engines (about 63%) and not the risk of an actual average engine. If no sensor raises the score meaningfully, the text says so and points to that reference instead of inventing a reason.
- The single-engine page says "Inspection recommended — above selected threshold" rather than ranking priority. Priority ordering is done in Fleet Priorities, using risk score and inspection capacity.
- Recommendations are templates keyed to the sensors the model actually flagged. They are worded as areas to verify, never as a diagnosis or a repair claim. Model explanation, possible areas to investigate, recommended action and "confirmed diagnosis: not available" are kept separate.
- Readings outside the training range show a reliability warning, the prediction itself is unchanged, and the engine is never labeled "routine monitoring" on the strength of a low score alone.
- Fleet Priorities: **predict every engine first, prioritise inspections only when needed.** Uploading a CSV scores every valid row (no capacity needed) and shows Total Engines Analyzed, Needs Inspection, Below Inspection Threshold and Percentage Flagged, a complete table (filter: All Engines / Needs Inspection / Below Inspection Threshold) and "Export all predictions". An optional section, "Optional: Prioritize Inspections", takes an inspection capacity and lists the highest-risk flagged engines first, with its own export; capacity never changes a score. An optional `Engine ID` column is carried through, otherwise the CSV row number identifies the engine. The all-predictions export has `risk_rank, engine_id, csv_row_number`, the six sensors, `predicted_probability`, `inspection_recommendation`, `main_model_driver`, `driver_odds_multiplier`, `review_focus` and `reliability_note`.

## Demo helpers

- **Three one-click examples** on the Predict page (above, near and below the default 0.50 threshold). They are real held-out test rows stored with the model, chosen and classified at run time by the trained model (`pickExamples` in `js/core.js`); nothing is hardcoded and the button captions show the model's own risk.
- **At a glance** card at the top of every result: risk, the main reason, and what to do next in about 60 words. The full explanation, sensor chart, next steps and limitations are one click below.
- **Reset demo** (top right) clears the readings and result, the uploaded fleet file, and restores the default threshold (0.50), cost assumptions and capacity.
- Everything runs offline: no external fonts, scripts or network calls. The page works from `node dashboard/serve.js` (no Python), and `demo.test.js` checks there are no remote resources.

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
node --test dashboard/tests/core.test.js dashboard/tests/explain.test.js dashboard/tests/demo.test.js
```

Covers the spline maths (partition of unity, linear extrapolation), input validation, CSV parsing, ranking, export, threshold and cost logic. Eighteen of the 47 tests need the real `model/model.json`: parity with pyGAM on 394 reference rows, contributions summing to the log-odds, saved metrics vs the project analysis (test AUC 0.7095, CV AUC 0.7041), and decision-table consistency. `tests/explain.test.js` covers the explanation and recommendation logic (see below). Without the model file the model-dependent tests are reported as skipped, not passed.

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
