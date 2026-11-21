// Run: node --test dashboard/tests   (Node 18+; no packages needed)
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const C = require('../js/core.js');

const close = (a, b, tol = 1e-9, msg) => assert.ok(Math.abs(a - b) <= tol, (msg || '') + ` expected ${b}, got ${a}`);

/* ---------- B-spline basis ---------- */
// Base knots 0..3 extended by the spacing on both sides, as scikit-learn does (6 basis functions).
const T = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6];
const DEG = 3;

test('spline basis sums to 1 inside the base interval (partition of unity)', () => {
  for (const x of [0, 0.3, 1, 1.5, 2.99, 3]) {
    const b = C.splineBasis(x, T, DEG);
    assert.equal(b.length, 6);
    close(b.reduce((s, v) => s + v, 0), 1, 1e-12, 'x=' + x);
    assert.ok(b.every((v) => v >= -1e-12));
  }
});

test('right end point of the base interval is evaluated on the last interval', () => {
  const b = C.splineBasis(3, T, DEG);
  close(b.reduce((s, v) => s + v, 0), 1, 1e-12);
  // continuity from the left
  const l = C.splineBasis(3 - 1e-9, T, DEG);
  b.forEach((v, i) => close(v, l[i], 1e-7));
});

test('linear extrapolation is continuous at the boundary and has the boundary slope', () => {
  for (const edge of [0, 3]) {
    const f = C.splineBasis(edge, T, DEG);
    const out = C.splineBasis(edge + (edge === 0 ? -1e-6 : 1e-6), T, DEG);
    f.forEach((v, i) => close(v, out[i], 1e-5));
    // linearity outside: second difference is zero
    const s = edge === 0 ? -1 : 1;
    const a = C.splineBasis(edge + s * 1, T, DEG), b = C.splineBasis(edge + s * 2, T, DEG), c = C.splineBasis(edge + s * 3, T, DEG);
    a.forEach((_, i) => close(a[i] - 2 * b[i] + c[i], 0, 1e-10));
    // slope outside equals the finite-difference slope just inside
    const h = 1e-6;
    const inside = C.splineBasis(edge - s * h, T, DEG);
    const slopeInside = f.map((v, i) => (v - inside[i]) / (s * h));
    const slopeOut = b.map((v, i) => (v - a[i]));
    slopeOut.forEach((v, i) => close(v * s, slopeInside[i], 1e-4));
  }
});

/* ---------- Tiny hand-made model ---------- */
function tinyArtifact() {
  const features = ['A', 'B'];
  const nS = 6;
  const coef = [0.5, -0.2, 0.1, 0.4, -0.3, 0.2, -0.1, 0.3, 0.2, -0.4, 0.1, 0.0];
  return {
    meta: {name: 'tiny'}, features,
    spline: {degree: DEG, n_splines: nS, knots: [T, T]},
    scaler: {mean: coef.map((_, i) => 0.1 + i * 0.01), scale: coef.map((_, i) => 0.5 + i * 0.05)},
    logistic: {coef, intercept: 0.3},
    ranges: {A: {min: 0, max: 3}, B: {min: 0, max: 3}}
  };
}

test('model: log-odds equals intercept plus the per-feature contributions, probability is the logistic of it', () => {
  const m = new C.Model(tinyArtifact());
  for (const row of [[1, 2], [0, 3], [-1, 5], [2.5, 0.2]]) {
    const parts = m.contributions(row);
    close(m.logit(row), 0.3 + parts[0] + parts[1], 1e-12);
    close(m.predict(row), 1 / (1 + Math.exp(-m.logit(row))), 1e-12);
  }
});

test('model rejects an inconsistent artifact', () => {
  const a = tinyArtifact(); a.logistic.coef.pop();
  assert.throws(() => new C.Model(a));
});

test('validation: missing, non-numeric, and out-of-range inputs', () => {
  const m = new C.Model(tinyArtifact());
  let v = C.validateRow(m, {A: '1', B: ''});
  assert.equal(v.ok, false); assert.match(v.errors[0], /B is missing/);
  v = C.validateRow(m, {A: 'abc', B: '2'});
  assert.equal(v.ok, false); assert.match(v.errors[0], /A must be a number/);
  v = C.validateRow(m, {A: 'Infinity', B: '2'});
  assert.equal(v.ok, false);
  v = C.validateRow(m, {A: '5', B: '2,5'});
  assert.equal(v.ok, true); assert.equal(v.values[1], 2.5); assert.equal(v.warnings.length, 1);
  v = C.validateRow(m, {A: 1, B: 2});
  assert.equal(v.ok, true); assert.equal(v.warnings.length, 0);
});

test('validation: negative RPM or pressure is rejected, negative temperature is only a range warning', () => {
  const a = tinyArtifact();
  a.units = ['rpm', 'degC'];
  const m = new C.Model(a);
  let v = C.validateRow(m, {A: '-5', B: '1'});
  assert.equal(v.ok, false); assert.match(v.errors[0], /A cannot be negative/);
  v = C.validateRow(m, {A: '1', B: '-5'});
  assert.equal(v.ok, true); assert.equal(v.warnings.length, 1);
});

/* ---------- Decision, capacity and cost maths ---------- */
const tab = {thresholds: [0, 0.5, 1], tp: [60, 50, 0], fp: [40, 20, 0], fn: [0, 10, 60], tn: [0, 20, 40]}; // n = 100, 60 positives

test('confusion and cost: known values', () => {
  const c = C.confusionAt(tab, 1);
  assert.equal(c.n, 100);
  close(c.precision, 50 / 70, 1e-12); close(c.recall, 50 / 60, 1e-12); close(c.flaggedShare, 0.7, 1e-12);
  close(C.costFor(c, 1), (20 + 10) / 100, 1e-12);
  close(C.costFor(c, 3), (20 + 30) / 100, 1e-12);
  const s = C.strategies(tab, 2, 1);
  close(s.all.cost, 40 / 100, 1e-12);       // inspect all: every healthy engine is an unnecessary inspection
  close(s.none.cost, (2 * 60) / 100, 1e-12); // inspect none: every case is missed
  close(s.model.cost, (20 + 2 * 10) / 100, 1e-12);
});

test('best threshold: moves toward flagging more engines as misses get costlier', () => {
  const r1 = C.bestThreshold(tab, 1), r10 = C.bestThreshold(tab, 10);
  assert.equal(r1.threshold, 0.5);
  assert.equal(r10.threshold, 0); // flag everyone
  assert.ok(C.thresholdIndex(tab.thresholds, 0.4) === 1 && C.thresholdIndex(tab.thresholds, 0.1) === 0);
});

test('capture and percentile helpers', () => {
  const cap = {pct: Array.from({length: 100}, (_, i) => i + 1), n: Array.from({length: 100}, (_, i) => i + 1), tp: Array.from({length: 100}, (_, i) => Math.round((i + 1) * 0.9))};
  const o = {prevalence: 0.6, n: 100};
  const r = C.captureAt(cap, o, 10);
  close(r.precision, 1 / 1 * (9 / 10), 1e-12); close(r.recall, 9 / 60, 1e-12);
  const q = Array.from({length: 101}, (_, i) => i / 100);
  close(C.scorePercentile(q, 0.5), 50, 1.1);
  assert.equal(C.scorePercentile(q, -1), 0); assert.equal(C.scorePercentile(q, 2), 100);
});

/* ---------- CSV ---------- */
test('CSV parser handles BOM, CRLF, quotes, and delimiters', () => {
  assert.deepEqual(C.parseCSV('﻿a,b\r\n1,"x,y"\r\n"he said ""hi""",3\r\n'), [['a', 'b'], ['1', 'x,y'], ['he said "hi"', '3']]);
  assert.deepEqual(C.parseCSV('a;b\n1;2\n'), [['a', 'b'], ['1', '2']]);
  assert.deepEqual(C.parseCSV('a\tb\n1\t2'), [['a', 'b'], ['1', '2']]);
  assert.deepEqual(C.parseCSV('a,b\n\n1,2\n'), [['a', 'b'], ['1', '2']]);
});

test('scoreCSV: ranks by risk, labels rows by CSV row number, skips invalid rows, ignores extra columns', () => {
  const m = new C.Model(tinyArtifact());
  const text = 'Extra,b,A\nx,2,1\ny,oops,1\nz,3,0.5\nw,,2\nq,1,2.5\n'; // header names are matched case/space-insensitively
  const res = C.scoreCSV(m, text);
  assert.equal(res.rows.length, 3);
  assert.deepEqual(res.invalid.map((x) => x.rowNumber), [3, 5]);
  assert.deepEqual(res.extraColumns, ['Extra']);
  for (let i = 1; i < res.rows.length; i++) assert.ok(res.rows[i - 1].risk >= res.rows[i].risk);
  res.rows.forEach((r, i) => assert.equal(r.rank, i + 1));
  const byRow = Object.fromEntries(res.rows.map((r) => [r.rowNumber, r]));
  close(byRow[2].risk, m.predict([1, 2]), 1e-12);
  close(byRow[4].risk, m.predict([0.5, 3]), 1e-12);
  close(byRow[6].risk, m.predict([2.5, 1]), 1e-12);
});

test('scoreCSV: missing columns and empty files give clear errors', () => {
  const m = new C.Model(tinyArtifact());
  assert.match(C.scoreCSV(m, 'A,C\n1,2\n').error, /Missing required column\(s\): B/);
  assert.match(C.scoreCSV(m, 'A,B\n').error, /at least one data row/);
});

test('exportCSV: round trip keeps ranks, row numbers, values and the recommendation logic', () => {
  const m = new C.Model(tinyArtifact());
  const res = C.scoreCSV(m, 'A,B\n1,2\n0.5,3\n2.5,1\n2,2\n');
  const thr = res.rows[1].risk; // flag the top two
  const text = C.exportCSV(m, res.rows, thr, 1);
  const grid = C.parseCSV(text);
  assert.deepEqual(grid[0], ['rank', 'csv_row_number', 'A', 'B', 'risk_score', 'above_threshold', 'within_capacity', 'recommended_action', 'main_model_driver', 'driver_odds_multiplier', 'review_focus', 'reliability_note']);
  assert.equal(grid.length, 5);
  assert.equal(grid[1][0], '1'); assert.equal(grid[1][6], 'yes'); assert.equal(grid[1][7], 'Inspect (human review)');
  assert.equal(grid[2][5], 'yes'); assert.equal(grid[2][6], 'no'); assert.match(grid[2][7], /over capacity/);
  assert.equal(grid[3][5], 'no'); assert.equal(grid[3][7], 'Routine monitoring');
  grid.slice(1).forEach((row, i) => { close(Number(row[4]), res.rows[i].risk, 1e-6); assert.equal(Number(row[1]), res.rows[i].rowNumber); });
});

/* ---------- The real exported model (skipped until notebooks/dashboard_model_export.ipynb has been run) ---------- */
const modelPath = path.join(__dirname, '..', 'model', 'model.json');
const real = fs.existsSync(modelPath) ? JSON.parse(fs.readFileSync(modelPath, 'utf8')) : null;
const skip = !real ? 'dashboard/model/model.json not found: run notebooks/dashboard_model_export.ipynb in Colab' : real.meta.fixture ? 'model.json is a development fixture' : false;

test('real model: JavaScript reproduces scikit-learn predictions on every reference row', {skip}, () => {
  const m = new C.Model(real);
  let worst = 0;
  real.reference.X.forEach((row, i) => { worst = Math.max(worst, Math.abs(m.predict(row) - real.reference.proba[i])); });
  assert.ok(real.reference.X.length >= 250);
  assert.ok(worst < 1e-8, 'max abs difference ' + worst);
});

test('real model: contributions add up to the log-odds', {skip}, () => {
  const m = new C.Model(real);
  for (const row of real.reference.X.slice(0, 50)) close(m.logit(row), m.intercept + m.contributions(row).reduce((s, v) => s + v, 0), 1e-12);
});

test('real model: saved metrics match the project evaluation', {skip}, () => {
  const t = real.metrics.test, cv = real.metrics.cv;
  assert.equal(real.meta.n_dev, 16604); assert.equal(t.n, 2931);
  close(t.auc, 0.7095, 0.0006, 'test AUC (project analysis, GAM: 0.7095)');
  assert.ok(t.auc_ci[0] > 0.68 && t.auc_ci[1] < 0.74);
  close(cv.auc_mean, 0.7041, 0.001, 'CV AUC (project analysis, GAM: 0.7041)');
  assert.ok(Number.isFinite(cv.auc_std) && Number.isFinite(cv.brier_mean), 'cross-validation metrics must be finite');
  assert.equal(cv.n_folds, 15);
  assert.match(real.meta.trained_on, /train \+ val only/);
});

test('real model: decision tables are internally consistent', {skip}, () => {
  const o = real.oof;
  const total = o.tp[0] + o.fp[0] + o.fn[0] + o.tn[0];
  assert.equal(total, o.n); assert.equal(o.n, 16604);
  o.thresholds.forEach((_, i) => {
    assert.equal(o.tp[i] + o.fp[i] + o.fn[i] + o.tn[i], o.n);
    assert.equal(o.tp[i] + o.fn[i], o.tp[0]);
    if (i) { assert.ok(o.tp[i] <= o.tp[i - 1]); assert.ok(o.fp[i] <= o.fp[i - 1]); }
  });
  assert.equal(real.capture.tp[99], o.tp[0]);
  assert.equal(real.oof.score_quantiles.length, 101);
  const tt = real.test_table; assert.equal(tt.tp[0] + tt.fp[0] + tt.fn[0] + tt.tn[0], 2931);
});
