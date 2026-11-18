// Run: node --test dashboard/tests/explain.test.js
// Tests for the explanation and recommendation logic. Logic tests use a small hand-made model;
// the "real model" tests use dashboard/model/model.json and are skipped (not passed) when it is absent.
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const C = require('../js/core.js');

const T = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6];
function tinyArtifact(coefScale = 1, intercept = 0.3) {
  const base = [0.5, -0.2, 0.1, 0.4, -0.3, 0.2, -0.1, 0.3, 0.2, -0.4, 0.1, 0.0];
  const coef = base.map((v) => v * coefScale);
  const r = {min: 0, p25: 1, median: 1.5, p75: 2, max: 3};
  return {
    meta: {name: 'tiny'}, features: ['Fuel Pressure', 'Lub Oil Temp'], units: ['bar', 'degC'],
    spline: {degree: 3, n_splines: 6, knots: [T, T]},
    scaler: {mean: coef.map((_, i) => 0.1 + i * 0.01), scale: coef.map((_, i) => 0.5 + i * 0.05)},
    logistic: {coef, intercept},
    ranges: {'Fuel Pressure': r, 'Lub Oil Temp': r}
  };
}

// Words that would turn a model score into a diagnosis or a repair claim.
const FORBIDDEN = /\b(replace|replacement|broken|defective|malfunction\w*|is failing|will fail|needs repair|has failed|worn out|pump is|bearing)\b/i;
const allText = (e) => [].concat(e.narrative, e.items.map((i) => i.sentence), e.steps.map((s) => s.text), e.priority).join(' \n ');

test('explanation is built from the actual model contributions and adds up to the log-odds', () => {
  const m = new C.Model(tinyArtifact(3));
  for (const row of [[1, 2], [0.2, 2.9], [2.5, 0.4]]) {
    const e = C.explainPrediction(m, row, 0.5);
    const c = m.contributions(row);
    e.items.forEach((it, j) => {
      assert.equal(it.contribution, c[j]);
      assert.ok(Math.abs(it.oddsFactor - Math.exp(c[j])) < 1e-12);
    });
    assert.ok(Math.abs(e.logit - (m.intercept + c[0] + c[1])) < 1e-12);
    assert.ok(Math.abs(e.p - m.predict(row)) < 1e-12);
    assert.ok(Math.abs(e.baseline - C.explainPrediction(m, row, 0.5).baseline) < 1e-12);
  }
});

test('a sensor is described as raising risk only when its contribution is meaningfully positive', () => {
  const m = new C.Model(tinyArtifact(3));
  for (let a = 0; a <= 3; a += 0.25) for (let b = 0; b <= 3; b += 0.25) {
    const e = C.explainPrediction(m, [a, b], 0.5);
    e.items.forEach((it) => {
      const raises = /raises the odds/.test(it.sentence), lowers = /lowers the odds/.test(it.sentence);
      assert.equal(raises, it.contribution >= C.MEANINGFUL, it.sentence);
      assert.equal(lowers, it.contribution <= -C.MEANINGFUL, it.sentence);
      assert.equal(it.direction === 'up', raises);
    });
    assert.ok(e.up.every((i) => i.contribution > 0));
    assert.ok(e.down.every((i) => i.contribution < 0));
    for (let i = 1; i < e.up.length; i++) assert.ok(e.up[i - 1].contribution >= e.up[i].contribution);
  }
});

test('no meaningful driver: the explanation points to baseline risk instead of inventing a reason', () => {
  const m = new C.Model(tinyArtifact(0.01, 0.8)); // tiny coefficients: no sensor matters
  const e = C.explainPrediction(m, [1, 2], 0.5);
  assert.equal(e.up.length, 0);
  assert.equal(e.areas.length, 0);
  const text = e.narrative.join(' ');
  assert.match(text, /No single sensor raises/);
  assert.match(text, /baseline risk of 69%/);
  assert.match(text, /baseline alone is above your current threshold/);
  assert.equal(C.mainDriver(m, [1, 2]), null);
  assert.match(C.reviewFocus(null), /No single sensor driver/);
});

test('levels and priority follow the threshold, contributions stay the same', () => {
  const m = new C.Model(tinyArtifact(3));
  const row = [1.2, 1.7];
  const p = m.predict(row);
  const hi = C.explainPrediction(m, row, p - 0.05), near = C.explainPrediction(m, row, p + 0.05), lo = C.explainPrediction(m, row, p + 0.3);
  assert.equal(hi.level, 'hi'); assert.match(hi.priority, /High priority/);
  assert.equal(near.level, 'mid'); assert.equal(near.priority, 'Watch list');
  assert.equal(lo.level, 'lo'); assert.equal(lo.priority, 'Routine monitoring');
  assert.deepEqual(hi.items.map((i) => i.contribution), lo.items.map((i) => i.contribution));
  assert.match(lo.actions.join(' '), /routine monitoring/i);
  assert.match(hi.actions.join(' '), /qualified technician/);
});

test('readings outside the training range: warning, no routine-monitoring advice, numeric prediction unchanged', () => {
  const m = new C.Model(tinyArtifact(3));
  const row = [9, 1.5]; // Fuel Pressure far above the training max of 3
  for (const thr of [0.05, 0.5, 0.95]) {
    const e = C.explainPrediction(m, row, thr);
    assert.equal(e.reliable, false);
    assert.deepEqual(e.outOfRange.map((i) => i.name), ['Fuel Pressure']);
    assert.ok(Math.abs(e.p - m.predict(row)) < 1e-15, 'prediction must not be altered');
    assert.notEqual(e.priority, 'Routine monitoring');
    assert.match(e.priority, /verify readings/i);
    assert.match(e.narrative.join(' '), /Reliability warning: Fuel Pressure is outside the range seen in training, so the model is extrapolating/);
    assert.match(e.caution, /human review/);
    assert.ok(!/Continue routine monitoring/.test(allText(e)));
  }
  const inRange = C.explainPrediction(m, [1, 1.5], 0.5);
  assert.equal(inRange.reliable, true); assert.equal(inRange.caution, null);
});

test('recommendations are context-sensitive and never claim a fault or a repair', () => {
  const m = new C.Model(tinyArtifact(3));
  let sawFuel = false, sawOil = false;
  for (let a = 0; a <= 3; a += 0.25) for (let b = 0; b <= 3; b += 0.25) for (const thr of [0.3, 0.5, 0.7]) {
    const e = C.explainPrediction(m, [a, b], thr);
    const text = allText(e);
    assert.ok(!FORBIDDEN.test(text), 'forbidden wording in: ' + text);
    // recommendations only for sensors that raise risk
    e.areas.forEach((x) => assert.ok(e.up.some((i) => i.name === x.feature)));
    e.up.forEach((i) => assert.ok(e.areas.some((x) => x.feature === i.name)));
    e.areas.forEach((x) => { if (x.feature === 'Fuel Pressure') { sawFuel = true; assert.match(x.text, /fuel-pressure measurement/); assert.match(x.text, /manufacturer/); } if (x.feature === 'Lub Oil Temp') { sawOil = true; assert.match(x.text, /lubricating-oil temperature reading/); } });
    assert.ok(e.steps.every((s) => ['caution', 'area', 'action'].includes(s.kind)));
  }
  assert.ok(sawFuel && sawOil, 'both sensors should have appeared as drivers somewhere in the grid');
});

test('fleet row action: out-of-range rows are never routine, over-capacity rows are scheduled', () => {
  const row = (o) => Object.assign({risk: 0.2, rank: 5, outOfRange: []}, o);
  assert.equal(C.fleetAction(row({risk: 0.9, rank: 1}), 0.5, 3), 'Inspect (human review)');
  assert.match(C.fleetAction(row({risk: 0.9, rank: 4}), 0.5, 3), /over capacity/);
  assert.equal(C.fleetAction(row({}), 0.5, null), 'Routine monitoring');
  assert.match(C.fleetAction(row({outOfRange: ['Coolant Temp']}), 0.5, null), /Verify readings/);
});

/* ---------- Real exported model ---------- */
const modelPath = path.join(__dirname, '..', 'model', 'model.json');
const real = fs.existsSync(modelPath) ? JSON.parse(fs.readFileSync(modelPath, 'utf8')) : null;
const skip = !real ? 'dashboard/model/model.json not found' : real.meta.fixture ? 'model.json is a development fixture' : false;

function realCases() {
  const m = new C.Model(real);
  const X = real.reference.X.slice(0, 250); // real held-out test rows, all inside the training range
  const ps = X.map((r) => m.predict(r));
  const idx = (f) => { let best = -1; ps.forEach((p, i) => { if (f(p, i) && (best < 0)) best = i; }); return best; };
  const order = ps.map((_, i) => i).sort((a, b) => ps[b] - ps[a]);
  return {m, X, ps, highest: order[0], lowest: order[order.length - 1], near: idx((p) => p >= 0.4 && p < 0.5)};
}

test('real model: high-risk row has clear positive contributors that match the model', {skip}, () => {
  const {m, X, highest} = realCases();
  const e = C.explainPrediction(m, X[highest], 0.5);
  assert.equal(e.level, 'hi'); assert.ok(e.reliable);
  assert.ok(e.up.length >= 1);
  const c = m.contributions(X[highest]);
  e.up.forEach((i) => assert.ok(c[i.index] >= C.MEANINGFUL));
  const maxIdx = c.indexOf(Math.max(...c));
  assert.equal(e.up[0].index, maxIdx);
  assert.equal(C.mainDriver(m, X[highest]).index, maxIdx);
  assert.match(e.narrative[0], /flagged for inspection/);
  assert.ok(e.steps.some((s) => /qualified technician/.test(s.text)));
});

test('real model: low-risk row is not flagged and is advised routine monitoring', {skip}, () => {
  const {m, X, lowest} = realCases();
  const e = C.explainPrediction(m, X[lowest], 0.5);
  assert.equal(e.level, 'lo'); assert.equal(e.priority, 'Routine monitoring');
  assert.match(e.actions.join(' '), /routine monitoring/i);
  assert.ok(e.down.length >= 1);
});

test('real model: a row close to the threshold is a watch-list case, and the threshold changes the level only', {skip}, () => {
  const {m, X, near} = realCases();
  assert.ok(near >= 0);
  const a = C.explainPrediction(m, X[near], 0.5), b = C.explainPrediction(m, X[near], 0.3), c = C.explainPrediction(m, X[near], 0.9);
  assert.equal(a.level, 'mid'); assert.equal(a.priority, 'Watch list');
  assert.equal(b.level, 'hi'); assert.equal(c.level, 'lo');
  assert.deepEqual(a.items.map((i) => i.contribution), c.items.map((i) => i.contribution));
  assert.equal(a.p, b.p);
});

test('real model: rows with several positive contributors list them in order of size', {skip}, () => {
  const {m, X} = realCases();
  const multi = X.map((r, i) => i).filter((i) => C.explainPrediction(m, X[i], 0.5).up.length >= 2);
  assert.ok(multi.length > 0, 'expected at least one row with 2+ positive drivers');
  const e = C.explainPrediction(m, X[multi[0]], 0.5);
  for (let i = 1; i < e.up.length; i++) assert.ok(e.up[i - 1].contribution >= e.up[i].contribution);
  assert.ok(e.areas.length === e.up.length);
  e.up.forEach((i) => assert.ok(e.narrative.join(' ').includes(i.name)));
});

test('real model: out-of-training-range rows get a reliability warning and keep the exact numeric prediction', {skip}, () => {
  const m = new C.Model(real);
  const edge = real.reference.X.slice(250);
  let n = 0;
  edge.forEach((row, k) => {
    const oor = C.outsideTraining(m, row);
    if (!oor.length) return;
    n++;
    for (const thr of [0.3, 0.5, 0.8]) {
      const e = C.explainPrediction(m, row, thr);
      assert.equal(e.reliable, false);
      assert.ok(Math.abs(e.p - real.reference.proba[250 + k]) < 1e-12, 'prediction must equal the scikit-learn reference');
      assert.notEqual(e.priority, 'Routine monitoring');
      assert.ok(!/Continue routine monitoring/.test(allText(e)));
      assert.match(e.narrative.join(' '), /Reliability warning/);
    }
  });
  assert.ok(n >= 12, 'expected many outside-range edge rows, got ' + n);
});

test('real model: no explanation anywhere contains a diagnosis or repair claim', {skip}, () => {
  const m = new C.Model(real);
  for (const row of real.reference.X) for (const thr of [0.2, 0.5, 0.65, 0.9]) {
    const text = allText(C.explainPrediction(m, row, thr));
    assert.ok(!FORBIDDEN.test(text), 'forbidden wording: ' + (text.match(FORBIDDEN) || [])[0]);
  }
});

test('real model: sentences agree with contributions on every reference row', {skip}, () => {
  const m = new C.Model(real);
  for (const row of real.reference.X) {
    const e = C.explainPrediction(m, row, 0.5);
    const c = m.contributions(row);
    e.items.forEach((it, j) => {
      assert.equal(it.contribution, c[j]);
      assert.equal(/raises the odds/.test(it.sentence), c[j] >= C.MEANINGFUL);
      assert.equal(/lowers the odds/.test(it.sentence), c[j] <= -C.MEANINGFUL);
    });
  }
});

test('real model: CSV export carries the main driver, review focus and reliability note', {skip}, () => {
  const m = new C.Model(real);
  const F = real.features;
  const rows = real.reference.X.slice(0, 20).concat(real.reference.X.slice(250, 254));
  const csv = F.join(',') + '\n' + rows.map((r) => r.join(',')).join('\n') + '\n';
  const res = C.scoreCSV(m, csv);
  const out = C.parseCSV(C.exportCSV(m, res.rows, 0.5, 5));
  const head = out[0];
  const col = (n) => head.indexOf(n);
  assert.ok(col('main_model_driver') > 0 && col('review_focus') > 0 && col('reliability_note') > 0);
  out.slice(1).forEach((g) => {
    const rn = Number(g[1]);
    const values = F.map((f) => Number(g[col(f)]));
    const d = C.mainDriver(m, values);
    assert.equal(g[col('main_model_driver')], d ? d.name : 'None (baseline risk)');
    assert.equal(g[col('review_focus')], C.reviewFocus(d));
    assert.ok(!FORBIDDEN.test(g.join(' ')));
    const oor = C.outsideTraining(m, values);
    assert.equal(g[col('reliability_note')], oor.length ? 'Outside training range: ' + oor.join('; ') : '');
    if (oor.length && Number(g[col('risk_score')]) < 0.5) assert.match(g[col('recommended_action')], /Verify readings/);
    assert.ok(rn >= 2);
  });
  assert.ok(out.slice(1).some((g) => g[col('reliability_note')] !== ''), 'out-of-range rows should be noted');
  assert.ok(out.slice(1).some((g) => g[col('main_model_driver')] !== 'None (baseline risk)'));
});
