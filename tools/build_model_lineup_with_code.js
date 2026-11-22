#!/usr/bin/env node
/**
 * Builds reports/SmartOps_Final_Colab_Analysis.html (results first, then all code) from the existing results page
 * (reports/smartops_models.html) plus a Code Appendix.
 *
 * The appendix is generated from the real source files in this repository (nothing is typed in by hand),
 * so the displayed code can never drift from the committed scripts and notebooks.
 *
 *   node tools/build_model_lineup_with_code.js
 *
 * Only Node.js is needed (no npm install). Idempotent: reads the base page and writes the new one.
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8').replace(/\r\n/g, '\n');
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const COLAB_URL = 'https://colab.research.google.com/github/cynthia-0316/smartops/blob/feat/final-colab-analysis/notebooks/SmartOps_Final_Colab_Analysis.ipynb';
const FINAL_NB = 'notebooks/SmartOps_Final_Colab_Analysis.ipynb';

/* ------------------------------------------------------------------ notebook helpers */
function nbCells(file) { return JSON.parse(read(file)).cells; }
const src = (c) => (Array.isArray(c.source) ? c.source.join('') : c.source);
function codeOf(file) {
  const cells = nbCells(file).filter((c) => c.cell_type === 'code' && src(c).trim());
  return cells.map((c, i) => `# ---------- cell ${i + 1} of ${cells.length} ----------\n${src(c).replace(/\n+$/, '')}`).join('\n\n') + '\n';
}
/* final notebook: split its code cells by the section heading that precedes them */
function finalNotebookBySection() {
  const out = {}; let cur = null; let n = {};
  for (const c of nbCells(FINAL_NB)) {
    const t = src(c);
    if (c.cell_type === 'markdown') {
      let m = t.match(/^## ([0A-M])\./);
      if (m) cur = m[1] === '0' ? 'M' : m[1];
      else if (/^## Export/.test(t)) cur = 'M';
    } else if (cur && t.trim()) {
      n[cur] = (n[cur] || 0) + 1;
      (out[cur] = out[cur] || []).push(`# ---------- cell ${n[cur]} ----------\n${t.replace(/\n+$/, '')}`);
    }
  }
  const res = {};
  for (const k in out) res[k] = out[k].join('\n\n') + '\n';
  return res;
}
const NB = finalNotebookBySection();

const F = (file, kind, label) => ({ file, kind, label: label || '', code: file.endsWith('.ipynb') ? codeOf(file) : read(file) });
const FINAL = (sec) => ({ file: FINAL_NB, kind: 'final', label: `code cells of Section ${sec}`, code: NB[sec] });
const HIST = 'historical';

/* ------------------------------------------------------------------ appendix content */
const SECTIONS = [
  { id: 'A', title: 'Data loading and dataset description', shows: 'Dataset shapes and the 19,535-row table', tiles: [
      'Downloads the three fixed splits (train 14,650 / validation 1,954 / test 2,931) from Hugging Face and checks they match the committed engine_data.csv.'],
    items: [FINAL('A'), F('scripts/download_data.py', HIST, 'dataset download helper used by the team')] },
  { id: 'B', title: 'Data cleaning: missing values, duplicates, outliers', shows: 'Data-quality results, outlier handling', tiles: [
      'Counts missing cells and duplicate rows, flags 1.5xIQR outliers per sensor without removing them, and keeps every row.'],
    items: [FINAL('B'), F('notebooks/engine_condition_modeling.ipynb', HIST, 'original cleaning, EDA, baseline and outlier-sensitivity notebook'),
            F('experiments/batch3.py', HIST, 'outlier winsorising and label-noise filtering experiments (no gain)')] },
  { id: 'C', title: 'Exploratory data analysis', shows: 'Risk curves, sensor independence', tiles: [
      'Class-conditional histograms, single-sensor AUCs and the Spearman correlation matrix (the sensors are close to independent).'],
    items: [FINAL('C')] },
  { id: 'D', title: 'Train / validation / test splitting and leakage prevention', shows: '"How to read the scores": the 15-fold protocol', tiles: [
      'Development set = train + validation (16,604 rows). Repeated stratified 5-fold CV, 3 repeats, seed 42. The test file can be opened only once.'],
    items: [FINAL('D'), F('src/common.py', HIST, 'shared protocol module: loaders, test-set guard, get_cv()'),
            F('experiments/harness.py', HIST, 'per-fold prediction harness behind experiments/log.csv'),
            F('notebooks/00_eval_protocol_demo.ipynb', HIST, 'protocol demo')] },
  { id: 'E', title: 'Baseline models', shows: 'Baselines family, straight-line logistic regression', tiles: [
      'Class-prior dummy model (AUC 0.50) and standardised logistic regression, scored on identical folds.'],
    items: [FINAL('E')] },
  { id: 'F', title: 'Machine learning model training', shows: 'All family cards, the 99-run record', tiles: [
      'Fresh reruns of one representative model per family, plus the original experiment scripts that produced the 99 saved configurations.'],
    items: [FINAL('F'), F('experiments/models.py', HIST, 'spline_lr() and KDE naive Bayes factories'),
            F('experiments/batch1.py', HIST, 'families: linear, spline LR, NB, EBM, kNN, forests, MLP, boosting references'),
            F('experiments/batch2.py', HIST, 'neural nets and boosting structure'),
            F('experiments/batch4.py', HIST, 'hidden-structure checks'),
            F('experiments/batch5.py', HIST, 'spline variants and penalised GAM'),
            F('experiments/torch_nets.py', HIST, 'PyTorch nets with piecewise-linear encoding'),
            F('notebooks/boosting_sklearn.ipynb', HIST, 'scikit-learn gradient boosting (#10)'),
            F('notebooks/boosting_lightgbm.ipynb', HIST, 'LightGBM (#12)'),
            F('notebooks/boosting_xgboost.ipynb', HIST, 'XGBoost (#11)'),
            F('notebooks/boosting_catboost.ipynb', HIST, 'CatBoost (#13)'),
            F('notebooks/feature_engineering.ipynb', HIST, 'feature sets (#14)')] },
  { id: 'G', title: 'Hyperparameter tuning', shows: 'Nested-CV scores (open vs filled dots)', tiles: [
      'Nested cross-validation: the search for knots and C runs inside each outer training fold, so the outer score is not flattered by tuning.'],
    items: [FINAL('G'), F('experiments/nested_final.py', HIST, 'nested CV of spline LR, CatBoost and the stack')] },
  { id: 'H', title: 'Cross-validation and model comparison', shows: 'Overview chart, 99-run table, test-of-difference claims', tiles: [
      'Compares the fresh CV scores with the saved runs (same folds and seeds), applies the corrected resampled t-test, and rebuilds the 99-run family summary.'],
    items: [FINAL('H'), F('notebooks/final_model_comparison.ipynb', HIST, 'results-log comparison and fold-by-fold tests (#18)'),
            F('experiments/collect_report_data.py', HIST, 'assembles the leaderboard behind the results pages')] },
  { id: 'I', title: 'Final model evaluation (held-out test set)', shows: 'Test-set results chart and table', tiles: [
      'Opens the test file once, scores the final spline LR, GAM and CatBoost, adds bootstrap intervals, ROC curves and a confusion matrix.'],
    items: [FINAL('I')] },
  { id: 'J', title: 'Feature importance and interpretability', shows: 'Risk curves for each sensor', tiles: [
      'Permutation importance (fit on train, shuffle on validation) and per-sensor response curves; no test data used.'],
    items: [FINAL('J'), F('notebooks/interpretation.ipynb', HIST, 'SHAP, permutation and partial dependence for CatBoost (#16)'),
            F('experiments/model_curves.py', HIST, 'representative-model risk curves shown on this page')] },
  { id: 'K', title: 'Threshold selection and business cost analysis', shows: 'Capacity view, hypothetical cost scenarios', tiles: [
      'Thresholds and costs are chosen on development out-of-fold predictions only; cost figures are hypothetical scenarios.'],
    items: [FINAL('K'), F('notebooks/threshold_and_calibration.ipynb', HIST, 'class weights, calibration, thresholds and cost view (#15)')] },
  { id: 'L', title: 'Final model selection and export', shows: 'Lessons: why spline LR is deployed', tiles: [
      'Refits the spline LR on all development data, saves it with joblib, exports knots and coefficients to JSON and verifies exact reload.'],
    items: [FINAL('L')] },
  { id: 'M', title: 'Reproduction instructions and dependencies', shows: 'Everything', tiles: [
      'Environment, install cell, verification checklist, provenance table and the in-Colab HTML export. Historical scripts: see experiments/RESULTS.md.'],
    items: [FINAL('M'), F('requirements.txt', HIST, 'project dependencies (pygam, interpret and torch are extra, see RESULTS.md)'),
            F('experiments/RESULTS.md', HIST, 'reproduce commands and results log (read-only reference)')] },
];

/* result section id on the page -> appendix sections that explain it */
const LINKS = {
  overview: ['H'], shapes: ['J'], reading: ['D', 'H'], 'f-base': ['E'], 'f-linear': ['E', 'F'], 'f-additive': ['F', 'G'],
  'f-nb': ['F'], 'f-knn': ['F'], 'f-trees': ['F'], 'f-boost': ['F', 'G'], 'f-nn': ['F'], 'f-ens': ['F', 'G'],
  tricks: ['B', 'K'], test: ['I', 'K'], lessons: ['L'], runs: ['H'], business: ['K', 'L'],
};

/* ------------------------------------------------------------------ transform the base page */
let page = read('reports/smartops_models.html');
function must(from, to, all) {
  if (!page.includes(from)) throw new Error('base page no longer contains: ' + from.slice(0, 70));
  page = all ? page.split(from).join(to) : page.replace(from, () => to);
}

must('<title>SmartOps Model Lineup</title>', '<title>SmartOps: Model Results, Python Code &amp; Reproducibility</title>');

/* --- wording: the deployed model is the spline LR; the GAM has the best single CV score; no superiority claims --- */
must('<li><b>Recommended model:</b> the GAM with 20 splines (CV 0.7041, test 0.7095). It is as accurate as anything else, fast, and you can plot exactly how each sensor moves the risk.</li>',
  '<li><b>Two deployable choices, no proven winner.</b> The GAM with 20 splines has the best single-model CV score (0.7041, test 0.7095) and chooses its own smoothing. The dashboard deploys the spline logistic regression (CV 0.7032 at its fixed setting, 0.7026 nested; test 0.7099) because it is transparent and light enough to run inside a browser. The gap between them is not significant (CV +0.0009, p = 0.21; test difference −0.0004, 95% CI −0.002 to +0.001), so a slightly higher test AUC is not evidence that one is better.</li>');
must('We picked the GAM because it is as accurate as anything else and chooses its own smoothness, so its score needed no extra tuning.',
  'The GAM has the best single-model CV score and chooses its own smoothness, so its score needed no extra tuning. The dashboard deploys the spline logistic regression instead: it is statistically tied with the GAM and lighter to run in a browser.');
must('"Recommended final model"', '"Best single CV model"');
must('recommended</span>', 'best CV</span>', true);
must('GAM (recommended)', 'GAM (best single CV score)');
must('<span><i class="sw hi"></i>Recommended model</span>', '<span><i class="sw hi"></i>Best CV score (GAM)</span>');
must('${esc(t.name)}${t.name.startsWith("GAM") ? ` <span class="tag best">best CV</span>` : ""}',
  '${esc(t.name)}${t.name.startsWith("GAM") ? ` <span class="tag best">best CV</span>` : ""}${t.name.startsWith("Spline") ? ` <span class="tag">deployed in dashboard</span>` : ""}');

/* --- nav, hero note, footer --- */
must('<a href="#runs">All runs</a>', '<a href="#runs">All runs</a><a href="#business">Business decisions</a><a href="#code-appendix" class="nav-code">Code Appendix</a>');
must('</p>\n  <div class="tiles">',
  '</p>\n  <p class="lede" style="font-size:0.95rem">Results come first. Below them, the <a href="#code-appendix">Code Appendix</a> holds the <b>executed Google Colab notebook</b> (every code cell with its real output and charts) and all the Python code behind the saved experiments. This file is the final submission version.</p>\n  <div class="tiles">');
/* --- hero: heading and subtitle say the page holds findings AND code; prominent jump links --- */
must('<h1>Every model we tried, and what each one taught us</h1>', '<h1>SmartOps: Model Results, Python Code &amp; Reproducibility</h1>');
must('<p class="lede">We ran 99 model configurations',
  '<p class="lede">Explore our model experiments, performance comparisons, interactive visualizations, and complete Python analysis with executed Google Colab results.</p>\n  <div class="cta"><a class="btn primary" href="#code-appendix">View the Python code &amp; executed Colab results ↓</a><a class="btn" href="#overview">Start with the model results</a></div>\n  <p class="lede" style="font-size:0.95rem">We ran 99 model configurations');
must('<span>Sources: the shared team results log', '<span>The Code Appendix is generated from the repository files by tools/build_model_lineup_with_code.js.</span>\n  <span>Sources: the shared team results log');

/* --- new results-side sections, placed before the 99-run table --- */
const business = `
<section id="business">
  <div class="sec-head">
    <span class="eyebrow">Why it matters</span>
    <h2>What this means for SmartOps</h2>
    <p>SmartOps helps maintenance teams decide which engines to inspect first when inspection capacity is limited. These experiments answer which kind of model is worth deploying for that job.</p>
  </div>
  <div class="card">
    <ul class="plain">
      <li><b>Interpretability and simple deployment beat a fraction of a percent.</b> Nine families land within about 0.002 AUC of each other, so the choice rests on other qualities. A spline logistic regression is a small set of knots and weights: a planner can inspect each sensor's curve, and the dashboard evaluates it in the browser with no server.</li>
      <li><b>Scores support inspection prioritisation. They are not diagnoses.</b> A high score means an engine resembles past engines labelled “needs maintenance”; it says nothing about which component is at fault.</li>
      <li><b>Cost-saving scenarios are hypothetical.</b> The data has no cost information. With cheap unneeded inspections the model can beat “inspect everything”; once a missed fault costs about three times an unneeded inspection, the cost-optimal rule flags almost every engine (see code appendix, Section K).</li>
      <li><b>Human review is required.</b> Roughly 20 to 30 percent of flagged engines are false alarms at the thresholds we examined (precision about 0.70 to 0.78 at thresholds 0.50 to 0.65), and an AUC near 0.71 is a ranking aid, not a verdict.</li>
      <li><b>Provenance is uncertain.</b> We do not know where the data came from, whether it is real or synthetic, or exactly how “needs maintenance” was defined, and there is no time or engine identifier. Treat every number here as a property of this dataset.</li>
    </ul>
  </div>
</section>

<section id="protocol">
  <div class="sec-head">
    <span class="eyebrow">Evidence and protocol</span>
    <h2>How to trust each number</h2>
  </div>
  <div class="card">
    <div class="grid2">
      <div><span class="lbl">Data separation</span><p>19,535 engines in total: 16,604 development rows (train + validation) for every model choice, and 2,931 held-out test rows opened once, after the choices were fixed. No threshold or setting was tuned on the test set.</p></div>
      <div><span class="lbl">Three kinds of CV score</span><p><b>Nested</b>: tuning happened inside each training fold (fair). <b>Fixed setting</b>: no tuning on these folds (fair). <b>Same-fold selection</b>: settings were chosen on the same folds (hollow dots, slightly optimistic). Never compare them as if they were the same.</p></div>
      <div><span class="lbl">Freshly executed vs saved results</span><p>The 99 configurations on this page are <b>saved, verified results</b> from the team's experiment log; they were not rerun (hours of compute). The final Colab notebook <b>re-executes</b> one representative model per family, a nested-CV spline LR, the test-set evaluation, importance and threshold analysis, and checks them against the saved numbers.</p></div>
      <div><span class="lbl">Which file is the official submission?</span><p><code>SmartOps_Final_Colab_Analysis.html</code>, exported from the executed Colab notebook, satisfies the course requirement (code plus executed results). This page is the presentation version with the same models and conclusions.</p></div>
    </div>
  </div>
</section>
`;
const decision = require('./decision_section.js')(path.join(ROOT, 'reports/colab_run/SmartOps_Final_Colab_Analysis.executed.ipynb'));
const businessFinal = business.replace(/<section id="business">[\s\S]*?<\/section>/, () => decision.html);
if (businessFinal === business) throw new Error('business section not replaced');
must('<section id="runs">', businessFinal + '\n<section id="runs">');

/* --- appendix: CSS, HTML, data, script --- */
const css = `
/* ---------- hero call-to-action ---------- */
.cta { display: flex; flex-wrap: wrap; gap: 10px; }
.cta .btn { font: 600 0.9rem/1 var(--f-body); text-decoration: none; padding: 11px 16px; border-radius: 8px; border: 1px solid var(--axis); background: var(--surface); color: var(--ink); }
.cta .btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
.cta .btn:hover { filter: brightness(0.96); }
nav.toc a.nav-code { color: var(--accent); font-weight: 600; background: var(--accent-soft); }
/* ---------- code appendix ---------- */
.codelink { margin-top: 2px; font-size: 0.82rem; color: var(--muted); }
.codelink a { font: 500 0.78rem/1 var(--f-mono); margin-left: 6px; text-decoration: none; border: 1px solid var(--axis); border-radius: 999px; padding: 4px 9px; display: inline-block; }
.codelink a:hover { background: var(--accent-soft); }
#code-appendix { --code-bg: #ffffff; --code-ink: #11161c; --tk-kw: #2a78d6; --tk-str: #eb6834; --tk-com: #6f7884; --tk-num: #8a5a00; --tk-fn: #11161c; --tk-bi: #2a78d6; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) #code-appendix { --code-bg: #0e1114; --code-ink: #eef1f4; --tk-kw: #4b93ea; --tk-str: #e0652f; --tk-com: #8f98a2; --tk-num: #e0a838; --tk-fn: #eef1f4; --tk-bi: #4b93ea; } }
:root[data-theme="dark"] #code-appendix { --code-bg: #0e1114; --code-ink: #eef1f4; --tk-kw: #4b93ea; --tk-str: #e0652f; --tk-com: #8f98a2; --tk-num: #e0a838; --tk-fn: #eef1f4; --tk-bi: #4b93ea; }
.app-bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.app-bar button, .copy-btn { font: inherit; font-size: 0.82rem; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--axis); background: var(--surface); color: var(--ink); cursor: pointer; }
.app-bar button:hover, .copy-btn:hover { background: var(--accent-soft); }
.app-bar .spacer { flex: 1; }
.provenance { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr)); gap: 10px; }
.provenance > div { border: 1px solid var(--grid); border-radius: 8px; padding: 10px 12px; display: grid; gap: 4px; align-content: start; }
.pill { display: inline-block; font: 600 0.68rem/1 var(--f-mono); padding: 4px 7px; border-radius: 4px; letter-spacing: 0.04em; white-space: nowrap; }
.pill.fresh { background: var(--accent-soft); color: var(--accent); }
.pill.hist { background: transparent; border: 1px solid var(--hi); color: var(--hi); }
details.sec { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; scroll-margin-top: 60px; }
details.sec > summary { list-style: none; cursor: pointer; padding: 14px 18px; display: grid; gap: 3px; }
details.sec > summary::-webkit-details-marker { display: none; }
details.sec > summary .t { font: 700 1.02rem/1.25 var(--f-display); display: flex; gap: 10px; align-items: baseline; }
details.sec > summary .t::before { content: "▸"; color: var(--muted); font-size: 0.85rem; transition: transform .15s; flex: none; }
details.sec[open] > summary .t::before { transform: rotate(90deg); }
details.sec > summary .s { font-size: 0.82rem; color: var(--ink-2); margin-left: 24px; }
details.sec .body { padding: 2px 18px 18px; display: grid; gap: 12px; }
.sec-note { font-size: 0.86rem; color: var(--ink-2); max-width: 80ch; }
.file { border: 1px solid var(--grid); border-radius: 8px; overflow: hidden; min-width: 0; }
.file-head { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; padding: 8px 10px; background: var(--plane); border-bottom: 1px solid var(--grid); }
.file-head .fn { font: 500 0.82rem/1.3 var(--f-mono); word-break: break-all; }
.file-head .lb { font-size: 0.78rem; color: var(--muted); flex: 1 1 200px; }
.file-head .meta { font: 400 0.72rem/1 var(--f-mono); color: var(--muted); white-space: nowrap; }
.file pre { margin: 0; background: var(--code-bg); color: var(--code-ink); font: 400 0.78rem/1.55 var(--f-mono); padding: 12px 14px; overflow: auto; max-height: 520px; tab-size: 4; -webkit-text-size-adjust: 100%; }
.file pre code { font: inherit; white-space: pre; }
.tk-kw { color: var(--tk-kw); font-weight: 600; } .tk-str { color: var(--tk-str); } .tk-com { color: var(--tk-com); font-style: italic; }
.tk-num { color: var(--tk-num); } .tk-fn { color: var(--tk-fn); } .tk-bi { color: var(--tk-bi); } .tk-dec { color: var(--tk-fn); }
.repro { font: 400 0.8rem/1.5 var(--f-mono); background: var(--code-bg); color: var(--code-ink); border: 1px solid var(--grid); border-radius: 8px; padding: 10px 12px; overflow-x: auto; white-space: pre; margin: 0; }
.nb-cells { display: grid; gap: 12px; }
.nb-md { display: grid; gap: 8px; }
.nb-md .nb-p, .nb-md ul, .nb-md ol { font-size: 0.9rem; color: var(--ink-2); }
.nb-h { margin-top: 8px; }
h3.nb-h { font-size: 1.15rem; } h4.nb-h { font-size: 1.05rem; } h5.nb-h { font-size: 0.95rem; }
.nb-out { padding: 8px 12px 12px; display: grid; gap: 8px; border-top: 1px solid var(--grid); min-width: 0; }
.nb-pre { margin: 0; font: 400 0.78rem/1.5 var(--f-mono); color: var(--ink); white-space: pre; overflow-x: auto; }
.nb-err { color: #b3261e; }
.nb-img { max-width: 100%; height: auto; border: 1px solid var(--grid); border-radius: 6px; background: #fff; }
.nb-table table { font-size: 0.8rem; } .nb-table td, .nb-table th { text-align: right; white-space: nowrap; } .nb-table th:first-child, .nb-table td:first-child { text-align: left; }
@media (max-width: 560px) { details.sec > summary { padding: 12px 12px; } details.sec .body { padding: 2px 10px 14px; } .file pre { font-size: 0.74rem; } }
`;
must('</style>', css + decision.css + '</style>');

const colabHtml = esc(COLAB_URL);
const NB_RENDERED = require('./render_notebook.js')(path.join(ROOT, 'reports/colab_run/SmartOps_Final_Colab_Analysis.executed.ipynb'));
const appendix = `
<section id="code-appendix">
  <div class="sec-head">
    <span class="eyebrow">Appendix</span>
    <h2>Code Appendix</h2>
    <p>Everything above can be traced to Python code. Sections A to M follow the order of the analysis. Each section shows the cells of the final Colab notebook (freshly executed) and, where relevant, the committed scripts and notebooks that produced the saved historical results. Nothing here was shortened or rewritten.</p>
  </div>
  <div class="card">
    <div class="provenance">
      <div><span class="pill fresh">FRESH</span><p>Re-executed in <code>SmartOps_Final_Colab_Analysis.ipynb</code> from the raw data: dataset checks, representative models of each family, nested CV of the spline LR, test evaluation, importance, thresholds and costs.</p></div>
      <div><span class="pill hist">HISTORICAL</span><p>Produced earlier by the scripts and notebooks shown with this tag. Their saved results (99 configurations, boosting test predictions) are loaded and verified, <b>not rerun</b>, in the final notebook.</p></div>
      <div><span class="lbl">Executed notebook, embedded below</span><p>All code cells with the outputs saved by Colab (tables, charts, 16/16 verification checks). Rendered from the executed notebook; nothing was rerun or edited.</p></div>
      <div><span class="lbl">Run it yourself</span><p><a href="${colabHtml}" target="_blank" rel="noopener">Open the notebook in Google Colab</a>, then Runtime &gt; Run all (about 15 to 25 minutes, no local install).</p></div>
    </div>
    <div class="app-bar">
      <button type="button" id="app-expand">Expand all</button>
      <button type="button" id="app-collapse">Collapse all</button>
      <span class="spacer"></span>
      <span class="note" id="app-count"></span>
    </div>
  </div>
  ${NB_RENDERED}
  <h3 style="margin-top:6px">Code by analysis section (A to M)</h3>
  <div id="app-list" style="display:grid;gap:10px"></div>
  <div class="card">
    <h3>Reproduction instructions</h3>
    <pre class="repro" tabindex="0">Colab (recommended, nothing to install locally)
  1. Open the notebook link above.
  2. Runtime &gt; Run all. The first cell installs pygam and catboost; data comes from Hugging Face,
     saved experiment artifacts from this GitHub repository.
  3. The last cell exports the executed notebook to SmartOps_Final_Colab_Analysis.html and downloads it.

Rebuild this page (Node.js only)
  node tools/build_model_lineup_with_code.js

Rerun the historical experiments (hours; Python venv with the extras listed in experiments/RESULTS.md)
  cd experiments
  python batch1.py   # then batch2..5, torch_nets.py, nested_final.py (each appends to log.csv)</pre>
    <p class="note">Secrets: none of these files contain tokens or credentials. The data is a public MIT-licensed Hugging Face dataset and the repository is public.</p>
  </div>
</section>
`;
must('</main>', appendix + '\n</main>');

const payload = SECTIONS.map((s) => ({
  id: s.id, title: s.title, shows: s.shows, tiles: s.tiles,
  items: s.items.map((i) => ({ file: i.file, kind: i.kind, label: i.label, code: i.code })),
}));
for (const s of payload) for (const i of s.items) if (!i.code || !i.code.trim()) throw new Error(`empty code for ${i.file} in section ${s.id}`);
const dataJson = JSON.stringify(payload).replace(/</g, '\\u003c');

const script = `
<script type="application/json" id="code-data">${dataJson}</script>
<script>
(function () {
  var SECTIONS = JSON.parse(document.getElementById("code-data").textContent);
  var LINKS = ${JSON.stringify(LINKS)};
  var KW = new Set("False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split(" "));
  var BI = new Set("print len range int float str list dict set tuple enumerate zip map filter sum min max abs round sorted open isinstance getattr setattr super type bool any all next iter reversed property staticmethod classmethod self cls display".split(" "));
  function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  /* tiny Python tokenizer: comments, (triple-quoted / prefixed) strings, decorators, numbers, keywords, builtins, calls */
  var RE = /(#[^\\n]*)|((?:[rRbBfFuU]{1,2})?(?:"""[\\s\\S]*?"""|'''[\\s\\S]*?'''|"(?:\\\\.|[^"\\\\\\n])*"|'(?:\\\\.|[^'\\\\\\n])*'))|(@[A-Za-z_][\\w.]*)|(\\b\\d[\\d_]*(?:\\.\\d+)?(?:[eE][+-]?\\d+)?\\b)|([A-Za-z_]\\w*)(?=\\s*\\()|([A-Za-z_]\\w*)|(%pip[^\\n]*)/g;
  function highlight(code) {
    var out = "", last = 0, m;
    RE.lastIndex = 0;
    while ((m = RE.exec(code))) {
      out += esc(code.slice(last, m.index)); last = RE.lastIndex;
      var t = m[0];
      if (m[1] !== undefined || m[7] !== undefined) out += '<span class="tk-com">' + esc(t) + "</span>";
      else if (m[2] !== undefined) out += '<span class="tk-str">' + esc(t) + "</span>";
      else if (m[3] !== undefined) out += '<span class="tk-dec">' + esc(t) + "</span>";
      else if (m[4] !== undefined) out += '<span class="tk-num">' + esc(t) + "</span>";
      else if (m[5] !== undefined) out += KW.has(t) ? '<span class="tk-kw">' + t + "</span>" : (BI.has(t) ? '<span class="tk-bi">' + t + "</span>" : '<span class="tk-fn">' + t + "</span>");
      else out += KW.has(t) ? '<span class="tk-kw">' + t + "</span>" : (BI.has(t) ? '<span class="tk-bi">' + t + "</span>" : esc(t));
    }
    return out + esc(code.slice(last));
  }
  function highlightOther(code, file) { return /\\.(py|ipynb)$/.test(file) ? highlight(code) : esc(code); }

  document.querySelectorAll(".nbcell").forEach(function (cell) {
    var code = cell.querySelector("code"), raw = code.textContent; code.innerHTML = highlight(raw);
    cell.querySelector(".copy-btn").addEventListener("click", function (e) {
      var b = e.currentTarget;
      function done(ok) { b.textContent = ok ? "Copied" : "Press Ctrl+C"; setTimeout(function () { b.textContent = "Copy code"; }, 1500); }
      function fallback() { var ta = document.createElement("textarea"); ta.value = raw; ta.style.position = "fixed"; ta.style.opacity = "0"; document.body.appendChild(ta); ta.select(); var ok = false; try { ok = document.execCommand("copy"); } catch (x) {} ta.remove(); done(ok); }
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(raw).then(function () { done(true); }, fallback); else fallback();
    });
  });
  var list = document.getElementById("app-list"), blocks = [];
  SECTIONS.forEach(function (s) {
    var d = document.createElement("details"); d.className = "sec"; d.id = "code-" + s.id;
    var nFiles = s.items.length;
    d.innerHTML = '<summary><span class="t">' + s.id + ". " + esc(s.title) + '</span><span class="s">Supports: ' + esc(s.shows) + " · " + nFiles + " code block" + (nFiles === 1 ? "" : "s") + "</span></summary>" +
      '<div class="body"><p class="sec-note">' + esc(s.tiles.join(" ")) + '</p><div class="files"></div></div>';
    var files = d.querySelector(".files"); files.style.display = "grid"; files.style.gap = "12px";
    s.items.forEach(function (it, idx) {
      var lines = it.code.split("\\n").length - 1;
      var f = document.createElement("div"); f.className = "file";
      f.innerHTML = '<div class="file-head"><span class="pill ' + (it.kind === "final" ? "fresh" : "hist") + '">' + (it.kind === "final" ? "FRESH" : "HISTORICAL") + '</span><span class="fn">' + esc(it.file) + '</span><span class="lb">' + esc(it.label) + '</span><span class="meta">' + lines + ' lines</span><button type="button" class="copy-btn" aria-label="Copy code of ' + esc(it.file) + '">Copy code</button></div><pre tabindex="0"><code></code></pre>';
      f.querySelector("code").dataset.pending = "1";
      blocks.push({ el: f.querySelector("code"), it: it });
      f.querySelector(".copy-btn").addEventListener("click", function (e) {
        var b = e.currentTarget;
        function done(ok) { b.textContent = ok ? "Copied" : "Press Ctrl+C"; setTimeout(function () { b.textContent = "Copy code"; }, 1500); }
        if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(it.code).then(function () { done(true); }, function () { fallback(); });
        else fallback();
        function fallback() { var ta = document.createElement("textarea"); ta.value = it.code; ta.style.position = "fixed"; ta.style.opacity = "0"; document.body.appendChild(ta); ta.select(); var ok = false; try { ok = document.execCommand("copy"); } catch (x) {} ta.remove(); done(ok); }
      });
      files.appendChild(f);
    });
    d.addEventListener("toggle", function () { if (d.open) render(d); });
    list.appendChild(d);
  });
  function render(d) { d.querySelectorAll("code[data-pending]").forEach(function (c) { var b = blocks.filter(function (x) { return x.el === c; })[0]; c.innerHTML = highlightOther(b.it.code, b.it.file); c.removeAttribute("data-pending"); }); }

  var total = SECTIONS.reduce(function (n, s) { return n + s.items.length; }, 0);
  document.getElementById("app-count").textContent = SECTIONS.length + " sections · " + total + " code blocks";
  document.getElementById("app-expand").addEventListener("click", function () { list.querySelectorAll("details").forEach(function (d) { d.open = true; }); });
  document.getElementById("app-collapse").addEventListener("click", function () { list.querySelectorAll("details").forEach(function (d) { d.open = false; }); });

  /* result -> code links */
  Object.keys(LINKS).forEach(function (id) {
    var sec = document.getElementById(id); if (!sec) return;
    var host = sec.querySelector(".card") || sec; var p = document.createElement("p"); p.className = "codelink";
    p.innerHTML = "Code for this result:" + LINKS[id].map(function (k) { return '<a href="#code-' + k + '" data-code="' + k + '">Section ' + k + " ↓</a>"; }).join("");
    host.appendChild(p);
  });
  function openFromHash() { var m = location.hash.match(/^#code-([A-M])$/); if (!m) return; var d = document.getElementById("code-" + m[1]); if (d) { d.open = true; d.scrollIntoView(); } }
  document.addEventListener("click", function (e) { var a = e.target.closest && e.target.closest("a[data-code]"); if (a) { var d = document.getElementById("code-" + a.dataset.code); if (d) d.open = true; } });
  window.addEventListener("hashchange", openFromHash); openFromHash();
})();
</script>
`;
must('<div id="tip" hidden></div>', '<div id="tip" hidden></div>');
page = page.replace(/<\/body>/, () => script + '</body>');

const outFile = path.join(ROOT, 'reports', 'SmartOps_Final_Colab_Analysis.html');
fs.writeFileSync(outFile, page);
console.log('wrote', path.relative(ROOT, outFile), (page.length / 1024).toFixed(0) + ' KB;',
  payload.length + ' sections,', payload.reduce((n, s) => n + s.items.length, 0) + ' code blocks,',
  payload.reduce((n, s) => n + s.items.reduce((m, i) => m + i.code.length, 0), 0) + ' chars of code');
