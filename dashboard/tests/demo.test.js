// Run: node --test dashboard/tests/demo.test.js
// Demo-readiness tests: one-click examples, the at-a-glance summary, Reset demo, offline operation.
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
const FORBIDDEN = /\b(replace|replacement|broken|defective|malfunction\w*|is failing|will fail|needs repair|has failed|worn out|pump is|bearing)\b/i;

const modelPath = path.join(__dirname, '..', 'model', 'model.json');
const real = fs.existsSync(modelPath) ? JSON.parse(fs.readFileSync(modelPath, 'utf8')) : null;
const skip = !real ? 'dashboard/model/model.json not found' : real.meta.fixture ? 'model.json is a development fixture' : false;

/* ---------- One-click examples ---------- */

test('pickExamples: above / near / below rows, classified by the model at the given threshold, in-range only', () => {
  const m = new C.Model(tinyArtifact(3));
  const rows = [];
  for (let a = 0; a <= 3; a += 0.25) for (let b = 0; b <= 3; b += 0.25) rows.push([a, b]);
  rows.push([9, 1.5]); // out of range: must never be picked
  for (const thr of [0.4, 0.5, 0.6]) {
    const ex = C.pickExamples(m, rows, thr);
    for (const k of ['hi', 'mid', 'lo']) {
      if (!ex[k]) continue;
      const p = m.predict(rows[ex[k].index]);
      assert.equal(ex[k].p, p);
      assert.equal(C.levelFor(p, thr), k);
      assert.equal(C.outsideTraining(m, rows[ex[k].index]).length, 0);
    }
  }
  const thr = 0.5, ex = C.pickExamples(m, rows, thr), tgt = {hi: thr + 0.3, mid: thr - 0.05, lo: thr - 0.25};
  for (const k of ['hi', 'mid', 'lo']) {
    if (!ex[k]) continue;
    rows.forEach((r) => {
      if (C.outsideTraining(m, r).length) return;
      const p = m.predict(r);
      if (C.levelFor(p, thr) === k) assert.ok(Math.abs(p - tgt[k]) >= ex[k].d - 1e-12, 'a nearer row exists for ' + k);
    });
  }
});

test('real model: the three demo examples are real test rows, picked and classified by the model (not hardcoded)', {skip}, () => {
  const m = new C.Model(real);
  const rows = real.reference.X.slice(0, real.reference.n_real_test_rows);
  const ex = C.pickExamples(m, rows, 0.5);
  assert.ok(ex.hi && ex.mid && ex.lo, 'all three examples must exist');
  assert.ok(ex.hi.p >= 0.5); assert.ok(ex.mid.p >= 0.4 && ex.mid.p < 0.5); assert.ok(ex.lo.p < 0.4);
  // independent re-derivation from the stored scikit-learn probabilities gives the same three rows
  const ref = real.reference.proba, thr = 0.5, tgt = {hi: 0.8, mid: 0.45, lo: 0.25};
  for (const k of ['hi', 'mid', 'lo']) {
    let best = -1, bd = Infinity;
    rows.forEach((r, i) => {
      if (C.outsideTraining(m, r).length) return;
      const p = ref[i], lv = p >= thr ? 'hi' : p >= thr - 0.1 ? 'mid' : 'lo';
      if (lv === k && Math.abs(p - tgt[k]) < bd) { bd = Math.abs(p - tgt[k]); best = i; }
    });
    assert.equal(ex[k].index, best, 'example ' + k + ' should match the scikit-learn based pick');
    assert.ok(Math.abs(ex[k].p - ref[best]) < 1e-12);
    const e = C.explainPrediction(m, rows[best], 0.5);
    assert.equal(e.level, k); assert.equal(e.reliable, true);
  }
  assert.equal(C.explainPrediction(m, rows[ex.hi.index], 0.5).priority, 'Inspection recommended — above selected threshold');
  assert.equal(C.explainPrediction(m, rows[ex.mid.index], 0.5).priority, 'Watch list');
  assert.equal(C.explainPrediction(m, rows[ex.lo.index], 0.5).priority, 'Routine monitoring');
});

/* ---------- At-a-glance summary ---------- */

test('at-a-glance summary: short, consistent with the contributions, no diagnosis wording', () => {
  const m = new C.Model(tinyArtifact(3));
  for (let a = 0; a <= 3; a += 0.5) for (let b = 0; b <= 3; b += 0.5) for (const thr of [0.3, 0.5, 0.7]) {
    const e = C.explainPrediction(m, [a, b], thr);
    const g = e.glance;
    assert.equal(g.headline, e.priority);
    assert.ok(g.reason.length > 20 && g.reason.length < 320, g.reason);
    assert.ok(g.next.length > 20 && g.next.length < 320, g.next);
    assert.ok(!FORBIDDEN.test(g.reason + ' ' + g.next));
    e.items.forEach((it) => {
      // a sensor is named as a reason to raise the score only when its contribution is meaningfully positive
      if (/^(Main reason|Closest concern)/.test(g.reason) && g.reason.includes(it.name + ' (')) assert.equal(it.direction, 'up', g.reason);
    });
    if (e.level === 'lo' && e.reliable) assert.match(g.next, /Keep routine monitoring/);
    if (e.level === 'hi' && e.reliable) assert.match(g.next, /inspection queue/);
    if (!e.reliable) assert.match(g.next, /^Verify the .* reading/);
  }
  const e = C.explainPrediction(m, [0.5, 2.5], 0.05);
  e.up.slice(0, 2).forEach((i) => assert.ok(e.glance.next.includes(i.name === 'Fuel Pressure' ? 'fuel-pressure reading' : 'lubricating-oil temperature reading')));
});

test('real model: glance for the three demo examples and an out-of-range engine reads correctly', {skip}, () => {
  const m = new C.Model(real);
  const rows = real.reference.X.slice(0, real.reference.n_real_test_rows);
  const ex = C.pickExamples(m, rows, 0.5);
  const hi = C.explainPrediction(m, rows[ex.hi.index], 0.5);
  assert.match(hi.glance.reason, /^Main reason: /); assert.match(hi.glance.next, /Start by checking the/);
  hi.up.slice(0, 2).forEach((i) => assert.ok(hi.glance.reason.includes(i.name)));
  const mid = C.explainPrediction(m, rows[ex.mid.index], 0.5).glance;
  assert.match(mid.next, /watch list/);
  const lo = C.explainPrediction(m, rows[ex.lo.index], 0.5).glance;
  assert.match(lo.reason, /pulls? it down from the model.s reference level \(65%\)/);
  assert.match(lo.next, /No extra inspection suggested/);
  const oor = C.explainPrediction(m, [5000, 2.5, 11, 3, 84, 81], 0.5).glance;
  assert.match(oor.reason, /outside the range seen in training/); assert.match(oor.next, /^Verify the Engine RPM reading first/);
});

test('real model: no glance text anywhere contains a diagnosis or repair claim', {skip}, () => {
  const m = new C.Model(real);
  for (const row of real.reference.X) for (const thr of [0.2, 0.5, 0.8]) {
    const g = C.explainPrediction(m, row, thr).glance;
    assert.ok(!FORBIDDEN.test(g.headline + ' ' + g.reason + ' ' + g.next));
    assert.ok(!/high[- ]priority/i.test(g.headline));
  }
});

/* ---------- Page structure: examples, reset, offline ---------- */

const dir = path.join(__dirname, '..');
const read = (f) => fs.readFileSync(path.join(dir, f), 'utf8');

test('demo controls exist: three examples built from the model, a Reset demo button, and reset restores defaults', () => {
  const html = read('index.html'), app = read('js/app.js');
  assert.match(html, /id="btn-reset"[^>]*>Reset demo</);
  assert.match(html, /id="examples"/);
  assert.ok(app.includes("['hi', 'Above threshold'], ['mid', 'Near threshold'], ['lo', 'Below threshold']"));
  assert.ok(app.includes('C.pickExamples(model, realRows, DEFAULTS.threshold)'));
  assert.ok(!/model says \d/.test(app), 'example buttons must not carry hardcoded risk numbers');
  const reset = app.slice(app.indexOf('function resetDemo'), app.indexOf('/* ---------- start'));
  ['threshold: DEFAULTS.threshold', 'ratio: DEFAULTS.ratio', 'unit: DEFAULTS.unit', 'capPct: DEFAULTS.capPct', 'fleet: null', 'lastRow: null', "$('csv').value = ''", "$('filter').value = 'all'", "$('cap').value = ''", 'capN: null', "$('curve-sensor').value = '0'", 'RESULT_PLACEHOLDER']
    .forEach((x) => assert.ok(reset.includes(x), 'reset must handle ' + x));
  assert.deepEqual(app.match(/const DEFAULTS = \{[^}]+\}/)[0], 'const DEFAULTS = {threshold: 0.5, ratio: 1, unit: 100, capPct: 10}');
});

test('the demo runs offline: no external resources, no network calls, no server-side code', () => {
  const html = read('index.html'), css = read('css/styles.css'), app = read('js/app.js'), core = read('js/core.js');
  assert.ok(!/(?:src|href)\s*=\s*["']\s*(?:https?:)?\/\//i.test(html), 'index.html references a remote file');
  assert.ok(!/url\(\s*["']?\s*(?:https?:)?\/\//i.test(css), 'css references a remote file');
  assert.ok(!/@import/i.test(css));
  for (const [name, src] of [['app.js', app], ['core.js', core]]) {
    assert.ok(!/\b(fetch|XMLHttpRequest|WebSocket|sendBeacon)\b/.test(src), name + ' makes a network call');
    assert.ok(!/\brequire\(|\bimport\s.*\sfrom\s/.test(src.replace(/typeof module[^\n]*\n/g, '')), name + ' needs a module loader');
  }
  const scripts = [...html.matchAll(/<script src="([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual(scripts, ['model/model_data.js', 'js/core.js', 'js/app.js']);
  scripts.forEach((s) => assert.ok(fs.existsSync(path.join(dir, s)) || !real, s + ' is missing'));
});

test('real model: model_data.js (loaded by the page) holds exactly the model.json content', {skip}, () => {
  const js = read('model/model_data.js');
  assert.ok(js.startsWith('window.SMARTOPS_MODEL = ') && js.trimEnd().endsWith(';'));
  const body = js.slice('window.SMARTOPS_MODEL = '.length).trimEnd().slice(0, -1);
  assert.deepEqual(JSON.parse(body), real);
});

test('Predict opens pre-filled with the model-classified Above threshold example, and Reset demo restores it', () => {
  const app = read('js/app.js');
  assert.ok(app.includes('defaultRow = picks.hi ? realRows[picks.hi.index] : null'), 'default engine is picked by the model, not hardcoded');
  assert.ok(!/prefillDefault[\s\S]{0,400}\b(\d{3,4}\.\d+|\[\s*\d)/.test(app.slice(app.indexOf('function prefillDefault'), app.indexOf('function buildForm'))), 'no hardcoded sensor values in the default');
  const start = app.slice(app.indexOf('/* ---------- start'));
  assert.ok(start.includes('prefillDefault();'), 'form is pre-filled on load');
  const reset = app.slice(app.indexOf('function resetDemo'), app.indexOf('/* ---------- start'));
  assert.ok(reset.includes('prefillDefault();'), 'Reset demo returns to the pre-filled state');
  assert.ok(app.includes('if (!state.skipScroll)'), 'pre-filling on load must not scroll the page');
});
