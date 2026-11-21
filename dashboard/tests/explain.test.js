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
    assert.ok(Math.abs(e.referenceProbability - 1 / (1 + Math.exp(-m.intercept))) < 1e-12);
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

test('no meaningful driver: the explanation points to the reference probability instead of inventing a reason', () => {
  const m = new C.Model(tinyArtifact(0.01, 0.8)); // tiny coefficients: no sensor matters
  const e = C.explainPrediction(m, [1, 2], 0.5);
  assert.equal(e.up.length, 0);
  assert.equal(e.areas.length, 0);
  const text = e.narrative.join(' ');
  assert.match(text, /No single sensor raises/);
  assert.match(text, /reference probability is 69%/);
  assert.match(text, /reference level alone is above your current threshold/);
  assert.ok(!/baseline/i.test(text), "must not call the intercept-only probability a baseline risk");
  assert.equal(C.mainDriver(m, [1, 2]), null);
  assert.match(C.reviewFocus(null), /No single sensor driver/);
});

test('levels and priority follow the threshold, contributions stay the same', () => {
  const m = new C.Model(tinyArtifact(3));
  const row = [1.2, 1.7];
  const p = m.predict(row);
  const hi = C.explainPrediction(m, row, p - 0.05), near = C.explainPrediction(m, row, p + 0.05), lo = C.explainPrediction(m, row, p + 0.3);
  assert.equal(hi.level, 'hi'); assert.equal(hi.priority, 'Inspection recommended — above selected threshold'); assert.ok(!/high priority/i.test(hi.priority));
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
    assert.equal(g[col('main_model_driver')], d ? d.name : 'None (no single sensor driver)');
    assert.equal(g[col('review_focus')], C.reviewFocus(d));
    assert.ok(!FORBIDDEN.test(g.join(' ')));
    const oor = C.outsideTraining(m, values);
    assert.equal(g[col('reliability_note')], oor.length ? 'Outside training range: ' + oor.join('; ') : '');
    if (oor.length && Number(g[col('risk_score')]) < 0.5) assert.match(g[col('recommended_action')], /Verify readings/);
    assert.ok(rn >= 2);
  });
  assert.ok(out.slice(1).some((g) => g[col('reliability_note')] !== ''), 'out-of-range rows should be noted');
  assert.ok(out.slice(1).some((g) => g[col('main_model_driver')] !== 'None (no single sensor driver)'));
});

/* ---------- Reference probability vs observed prevalence; no objective "High priority" label ---------- */
const sig = (z) => 1 / (1 + Math.exp(-z));

test('reference probability is the intercept-only probability and is explained as such, not as observed average risk', () => {
  const a = tinyArtifact(3, 0.8);
  a.metrics = {dev_prevalence: 0.5};
  const m = new C.Model(a);
  const e = C.explainPrediction(m, [1, 1.5], 0.5);
  assert.ok(Math.abs(e.referenceProbability - sig(0.8)) < 1e-12);
  assert.equal(e.observedPrevalence, 0.5);
  assert.match(e.referenceNote, /reference probability is 69%/);
  assert.match(e.referenceNote, /every sensor.s effect sits at its average in the development data/);
  assert.match(e.referenceNote, /not the observed share of development engines that carry the label \(50%\)/);
  assert.match(e.referenceNote, /not the risk of an actual "average" engine/);
  // without a stored prevalence the note must not invent one
  const noPrev = C.explainPrediction(new C.Model(tinyArtifact(3, 0.8)), [1, 1.5], 0.5);
  assert.equal(noPrev.observedPrevalence, null);
  assert.ok(!/\d+%\)/.test(noPrev.referenceNote));
});

test('real model: reference 65% and observed prevalence 63% are reported as different quantities', {skip}, () => {
  const m = new C.Model(real);
  const e = C.explainPrediction(m, real.reference.X[0], 0.5);
  assert.ok(Math.abs(e.referenceProbability - sig(real.logistic.intercept)) < 1e-12);
  assert.ok(Math.abs(e.referenceProbability - 0.6522) < 0.0005);
  assert.ok(Math.abs(e.observedPrevalence - real.metrics.dev_prevalence) < 1e-12);
  assert.ok(Math.abs(e.observedPrevalence - 0.6305) < 0.0005);
  assert.ok(e.referenceProbability - e.observedPrevalence > 0.01, 'they are different numbers');
  assert.match(e.referenceNote, /65%/); assert.match(e.referenceNote, /63%/);
  assert.match(e.referenceNote, /modeling reference, not the observed share/);
});

test('real model: wording never presents the intercept-only probability as observed or average engine risk', {skip}, () => {
  const m = new C.Model(real);
  const BAD = /(baseline risk|average engine|average development engine|observed average|average risk|typical engine.s risk)/i;
  for (const row of real.reference.X) for (const thr of [0.2, 0.5, 0.65, 0.9]) {
    const e = C.explainPrediction(m, row, thr);
    const text = allText(e) + ' ' + e.referenceNote;
    // The only allowed use of the phrase "average engine" is the explicit denial inside the reference note.
    const stripped = text.replace('not the risk of an actual "average" engine', '');
    assert.ok(!BAD.test(stripped), 'misleading baseline wording: ' + (stripped.match(BAD) || [])[0]);
    assert.match(e.referenceNote, /reference probability is 65%/);
  }
});

test('single-engine labels never call an engine objectively "High priority"', () => {
  const m = new C.Model(tinyArtifact(3));
  const allowed = new Set([
    'Inspection recommended — above selected threshold',
    'Inspection recommended — above selected threshold (verify readings first)',
    'Verify readings before routine monitoring', 'Watch list', 'Routine monitoring'
  ]);
  for (let a = -1; a <= 5; a += 0.5) for (const thr of [0.1, 0.5, 0.9]) {
    const e = C.explainPrediction(m, [a, 1.5], thr);
    assert.ok(allowed.has(e.priority), 'unexpected label: ' + e.priority);
    assert.ok(!/high[- ]priority/i.test(allText(e)));
  }
  assert.match(C.explainPrediction(m, [1.2, 1.7], 0.01).actions[0], /Fleet Priorities to rank it against other engines by risk score/);
});

test('the page source uses the corrected wording (no High priority, no average-engine baseline claim)', () => {
  const dir = path.join(__dirname, '..');
  const src = ['js/app.js', 'js/core.js', 'index.html'].map((f) => fs.readFileSync(path.join(dir, f), 'utf8')).join('\n');
  assert.ok(!/high[- ]priority/i.test(src), 'a High priority label is still in the source');
  assert.ok(!/average development engine/i.test(src));
  assert.ok(!/\(baseline\) \+/i.test(src));
  assert.ok(/Inspection recommended — above selected threshold|Inspection recommended — above selected threshold/.test(src));
  assert.ok(/E\.referenceNote/.test(fs.readFileSync(path.join(dir, 'js/app.js'), 'utf8')));
});

test('fleet ranking still orders by risk score and applies inspection capacity', () => {
  const m = new C.Model(tinyArtifact(3));
  const res = C.scoreCSV(m, 'Fuel Pressure,Lub Oil Temp\n0.5,0.5\n2.5,2.5\n1.2,1.7\n2,0.2\n');
  for (let i = 1; i < res.rows.length; i++) assert.ok(res.rows[i - 1].risk >= res.rows[i].risk);
  const thr = 0; // everything is above the threshold; capacity decides who is inspected now
  const grid = C.parseCSV(C.exportCSV(m, res.rows, thr, 2));
  const col = grid[0].indexOf('recommended_action');
  assert.equal(grid[1][col], 'Inspect (human review)'); assert.equal(grid[2][col], 'Inspect (human review)');
  assert.match(grid[3][col], /over capacity/);
});
