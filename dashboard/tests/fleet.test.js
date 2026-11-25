// Run: node --test dashboard/tests/fleet.test.js
// Fleet workflow: predict every engine first; inspection capacity is an optional, separate step.
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const C = require('../js/core.js');

const modelPath = path.join(__dirname, '..', 'model', 'model.json');
const real = fs.existsSync(modelPath) ? JSON.parse(fs.readFileSync(modelPath, 'utf8')) : null;
const skip = !real ? 'dashboard/model/model.json not found' : real.meta.fixture ? 'model.json is a development fixture' : false;

// Deterministic generator so the 500-row fleet is the same on every run.
function rng(seed) { let s = seed >>> 0; return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; }; }

function fleetCSV(model, n, withId, seed = 7) {
  const r = rng(seed), cols = model.features;
  const head = (withId ? ['Engine ID'] : []).concat(cols).join(',');
  const lines = [head];
  for (let i = 0; i < n; i++) {
    const vals = cols.map((f) => { const g = model.a.ranges[f]; return (g.p01 + r() * (g.p99 - g.p01)).toFixed(4); });
    lines.push((withId ? ['ENG-' + String(i + 1).padStart(4, '0')] : []).concat(vals).join(','));
  }
  return lines.join('\n') + '\n';
}

test('real model: a CSV with 500 engines produces 500 predictions, all computed by the trained model', {skip}, () => {
  const m = new C.Model(real);
  const text = fleetCSV(m, 500, true);
  const res = C.scoreCSV(m, text);
  assert.equal(res.error, undefined);
  assert.equal(res.rows.length, 500); assert.equal(res.invalid.length, 0);
  assert.equal(res.idColumn, 'Engine ID');
  res.rows.forEach((r) => {
    assert.ok(Math.abs(r.risk - m.predict(r.values)) < 1e-15, 'score must equal the model prediction for that row');
    assert.ok(r.risk > 0 && r.risk < 1);
  });
  // sorted by risk, ranks 1..500, every row keeps its own identifier
  for (let i = 1; i < 500; i++) assert.ok(res.rows[i - 1].risk >= res.rows[i].risk);
  assert.deepEqual(res.rows.map((r) => r.rank), Array.from({length: 500}, (_, i) => i + 1));
  assert.equal(new Set(res.rows.map((r) => r.id)).size, 500);
  res.rows.forEach((r) => assert.equal(r.id, 'ENG-' + String(r.rowNumber - 1).padStart(4, '0')));
});

test('real model: flagged and below-threshold counts always add up to the total, and follow the threshold', {skip}, () => {
  const m = new C.Model(real);
  const res = C.scoreCSV(m, fleetCSV(m, 500, true));
  let prev = Infinity;
  for (const thr of [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]) {
    const S = C.fleetSummary(res.rows, thr);
    assert.equal(S.total, 500); assert.equal(S.needs + S.below, 500);
    assert.equal(S.needs, res.rows.filter((r) => r.risk >= thr).length);
    assert.ok(Math.abs(S.pctFlagged - S.needs / 500) < 1e-12);
    assert.ok(S.needs <= prev, 'raising the threshold can only flag fewer engines'); prev = S.needs;
    res.rows.forEach((r) => assert.equal(C.recommendation(r, thr), r.risk >= thr ? 'Needs Inspection' : 'Below Inspection Threshold'));
  }
  assert.ok(C.fleetSummary(res.rows, 0.05).needs > C.fleetSummary(res.rows, 0.95).needs, 'the threshold must change the counts');
});

test('real model: inspection capacity is optional and never changes the predictions', {skip}, () => {
  const m = new C.Model(real);
  const res = C.scoreCSV(m, fleetCSV(m, 500, true));
  const before = res.rows.map((r) => [r.rowNumber, r.risk, r.rank]);
  const csvBefore = C.exportAllCSV(m, res.rows, 0.5);
  const flagged = C.fleetSummary(res.rows, 0.5).needs;
  for (const cap of [1, 30, flagged - 1, flagged, flagged + 50, 100000]) {
    const P = C.prioritize(res.rows, 0.5, cap);
    assert.equal(P.flagged, flagged);
    assert.equal(P.selected.length, Math.min(cap, flagged));
    assert.equal(P.outside, flagged - P.selected.length);
    assert.equal(P.spare, Math.max(0, cap - flagged));
    // the selected engines are exactly the highest-risk flagged ones
    P.selected.forEach((r, k) => { assert.ok(r.risk >= 0.5); if (k) assert.ok(P.selected[k - 1].risk >= r.risk); });
    const minSel = P.selected.length ? P.selected[P.selected.length - 1].risk : Infinity;
    res.rows.filter((r) => r.risk >= 0.5 && !P.selected.includes(r)).forEach((r) => assert.ok(r.risk <= minSel));
  }
  assert.deepEqual(res.rows.map((r) => [r.rowNumber, r.risk, r.rank]), before, 'scores and ranks must not change');
  assert.equal(C.exportAllCSV(m, res.rows, 0.5), csvBefore, 'the full prediction export must not depend on capacity');
  // no capacity entered: nothing is prioritised, the prediction workflow is unaffected
  for (const none of [NaN, null, 0, -3]) assert.equal(C.prioritize(res.rows, 0.5, none).selected.length, 0);
});

test('real model: both exports are complete, consistent and parse back', {skip}, () => {
  const m = new C.Model(real);
  const res = C.scoreCSV(m, fleetCSV(m, 500, true));
  const thr = 0.5, cap = 30;
  const all = C.parseCSV(C.exportAllCSV(m, res.rows, thr));
  assert.deepEqual(all[0].slice(0, 3), ['risk_rank', 'engine_id', 'csv_row_number']);
  assert.equal(all.length, 501);
  const col = (g, name) => g[0].indexOf(name);
  const iP = col(all, 'predicted_probability'), iR = col(all, 'inspection_recommendation');
  let needs = 0;
  all.slice(1).forEach((line, k) => {
    assert.equal(Number(line[0]), k + 1);
    assert.ok(Math.abs(Number(line[iP]) - res.rows[k].risk) < 1e-6);
    assert.equal(line[iR], res.rows[k].risk >= thr ? 'Needs Inspection' : 'Below Inspection Threshold');
    assert.ok(!/healthy|safe/i.test(line[iR]));
    if (line[iR] === 'Needs Inspection') needs++;
  });
  assert.equal(needs, C.fleetSummary(res.rows, thr).needs);
  const prio = C.parseCSV(C.exportPriorityCSV(m, res.rows, thr, cap));
  assert.deepEqual(prio[0].slice(0, 4), ['inspection_priority', 'risk_rank', 'engine_id', 'csv_row_number']);
  assert.equal(prio.length - 1, Math.min(cap, needs));
  prio.slice(1).forEach((line, k) => {
    assert.equal(Number(line[0]), k + 1);
    assert.equal(line[col(prio, 'inspection_recommendation')], 'Needs Inspection');
    // same engine, same probability as in the complete export
    const match = all.find((a) => a[1] === line[2]);
    assert.ok(match && Math.abs(Number(match[iP]) - Number(line[col(prio, 'predicted_probability')])) < 1e-12);
  });
  assert.equal(C.parseCSV(C.exportPriorityCSV(m, res.rows, thr, 0)).length, 1, 'no capacity: header only');
});

test('engine identifiers: original ID kept when present, CSV row reference otherwise; ID column is not a model input', () => {
  const T = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6];
  const r = {min: 0, p25: 1, median: 1.5, p75: 2, max: 3};
  const m = new C.Model({
    meta: {name: 'tiny'}, features: ['A', 'B'], units: ['u', 'u'], spline: {degree: 3, n_splines: 6, knots: [T, T]},
    scaler: {mean: new Array(12).fill(0.1), scale: new Array(12).fill(1)},
    logistic: {coef: [0.5, -0.2, 0.1, 0.4, -0.3, 0.2, -0.1, 0.3, 0.2, -0.4, 0.1, 0.0], intercept: 0.3}, ranges: {A: r, B: r}
  });
  const withId = C.scoreCSV(m, 'unit id,A,B\nX-9,1,2\n,0.5,3\nX-7,2,1\n');
  assert.equal(withId.idColumn, 'unit id'); assert.deepEqual(withId.extraColumns, []);
  assert.deepEqual(withId.rows.map((x) => x.id).sort(), ['', 'X-7', 'X-9']);
  const noId = C.scoreCSV(m, 'A,B,Notes\n1,2,hello\n');
  assert.equal(noId.idColumn, null); assert.deepEqual(noId.extraColumns, ['Notes']); assert.equal(noId.rows[0].id, '');
  // the identifier never influences the score
  assert.equal(C.scoreCSV(m, 'Engine ID,A,B\nzzz,1,2\n').rows[0].risk, C.scoreCSV(m, 'A,B\n1,2\n').rows[0].risk);
  const csv = C.parseCSV(C.exportAllCSV(m, withId.rows, 0.5));
  assert.ok(csv.slice(1).some((l) => l[1] === 'X-9') && csv.slice(1).some((l) => l[1] === '' && l[2] === '3'), 'blank ID keeps the CSV row number');
});

test('bad input: missing columns, empty file, and invalid rows are reported while valid rows are kept', () => {
  const T = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6];
  const r = {min: 0, p25: 1, median: 1.5, p75: 2, max: 3};
  const m = new C.Model({
    meta: {name: 'tiny'}, features: ['A', 'B'], units: ['u', 'u'], spline: {degree: 3, n_splines: 6, knots: [T, T]},
    scaler: {mean: new Array(12).fill(0.1), scale: new Array(12).fill(1)},
    logistic: {coef: new Array(12).fill(0.1), intercept: 0}, ranges: {A: r, B: r}
  });
  assert.match(C.scoreCSV(m, 'A,C\n1,2\n').error, /Missing required column\(s\): B/);
  assert.match(C.scoreCSV(m, 'A,B\n').error, /at least one data row/);
  assert.match(C.scoreCSV(m, '').error, /header row/);
  const res = C.scoreCSV(m, 'A,B\n1,2\nabc,2\n1,\n2,1\n');
  assert.equal(res.rows.length, 2); assert.deepEqual(res.invalid.map((x) => x.rowNumber), [3, 4]);
  assert.equal(C.fleetSummary([], 0.5).total, 0); assert.equal(C.fleetSummary([], 0.5).pctFlagged, 0);
});

/* ---------- Page structure ---------- */

const dir = path.join(__dirname, '..');
const read = (f) => fs.readFileSync(path.join(dir, f), 'utf8');

test('Fleet page: capacity is not required to analyze, and prediction and prioritisation are separate sections', () => {
  const html = read('index.html'), app = read('js/app.js');
  const fleet = html.slice(html.indexOf('id="fleet"'), html.indexOf('id="insights"'));
  // the capacity field is only inside the optional section, which sits after the prediction results
  assert.ok(fleet.indexOf('Optional: Prioritize Inspections') > fleet.indexOf('Prediction results for all engines'));
  assert.ok(fleet.indexOf('id="cap"') > fleet.indexOf('Optional: Prioritize Inspections'));
  assert.ok(!/id="cap"[^>]*\brequired\b/.test(fleet));
  ['All Engines', 'Needs Inspection', 'Below Inspection Threshold', 'btn-export-all', 'btn-export-prio', 'btn-sample-load', 'btn-sample'].forEach((s) => assert.ok(fleet.includes(s), s));
  // loading a file scores everything without reading the capacity field
  const load = app.slice(app.indexOf('function loadFleetText'), app.indexOf("$('btn-sample-load')"));
  assert.ok(!load.includes('capN') && !load.includes("$('cap')"), 'analysis must not depend on capacity');
  // no wording that calls below-threshold engines healthy or safe
  const text = fleet + app.slice(app.indexOf('function renderFleet'), app.indexOf('function download'));
  assert.ok(!/[>"' ](healthy|safe)\b[^.]*engine/i.test(text.replace(/not (certified )?(healthy|safe)[^.]*\./gi, '').replace(/not cleared as safe/gi, '')), 'below-threshold engines must not be called healthy or safe');
});

test('Fleet tables: every engine row can be opened to show the model explanation (same logic as the Predict page)', () => {
  const app = read('js/app.js');
  assert.ok(app.includes('function fleetDetail(r, cols)'), 'detail renderer exists');
  assert.ok(app.includes('C.explainPrediction(model, r.values, state.threshold)'), 'detail uses the real model explanation, not separate text');
  assert.ok(app.includes('data-open-predict') && app.includes('function openInPredict'), 'hand-off to the Predict page');
  assert.ok(/class="rowbtn"[^`]*aria-expanded/.test(app), 'rows are keyboard-accessible buttons with aria-expanded');
  assert.ok(app.includes("['fleet-table', 'prio-table']"), 'both the all-engines and the prioritised table are clickable');
  assert.ok(app.includes('fleetDetail(r, 6)') && app.includes('fleetDetail(r, 7)'), 'detail rows span the full table width');
  const detail = app.slice(app.indexOf('function fleetDetail'), app.indexOf('function openInPredict'));
  assert.match(detail, /not a fault diagnosis/);
  assert.ok(!/\b(replace|broken|defective|malfunction\w*|will fail)\b/i.test(detail));
});
