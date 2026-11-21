/* SmartOps core: model inference, decision/cost maths and CSV handling.
 * No dependencies. Works in the browser (window.SmartOpsCore) and in Node (module.exports).
 *
 * The model is an additive spline model exported from Python (see the notebook named in
 * model.json). Per sensor it stores a B-spline knot vector, the coefficients and the
 * development mean of each basis column, so the sensor effects (centred log-odds) add up
 * exactly to the logit. Inference re-implements the cubic B-spline basis with linear
 * extrapolation outside the training range, and is tested against the training library's
 * own predictions (dashboard/tests). Works for the penalised GAM (pyGAM) and for a
 * spline logistic regression alike.
 */
(function (root) {
  'use strict';

  /* ---------- B-spline basis (matches sklearn SplineTransformer) ---------- */

  // All B-spline basis functions of the given degree at x, using the knot interval `span`
  // (t[span] <= x <= t[span+1]). Passing the span explicitly keeps the right end point valid.
  function fullBasis(x, t, deg, span) {
    const m = t.length;
    let prev = new Array(m - 1).fill(0);
    prev[span] = 1;
    for (let d = 1; d <= deg; d++) {
      const cur = new Array(m - d - 1).fill(0);
      for (let j = 0; j < cur.length; j++) {
        const a = t[j + d] - t[j];
        const b = t[j + d + 1] - t[j + 1];
        const left = a > 0 ? ((x - t[j]) / a) * prev[j] : 0;
        const right = b > 0 ? ((t[j + d + 1] - x) / b) * prev[j + 1] : 0;
        cur[j] = left + right;
      }
      prev = cur;
    }
    return prev;
  }

  function spanFor(x, t, deg) {
    const nBasis = t.length - deg - 1;
    let i = deg;
    while (i < nBasis - 1 && t[i + 1] <= x) i++;
    return i;
  }

  function basisInside(x, t, deg) {
    return fullBasis(x, t, deg, spanFor(x, t, deg));
  }

  function basisDerivative(x, t, deg) {
    const nBasis = t.length - deg - 1;
    const span = spanFor(x, t, deg);
    const lower = fullBasis(x, t, deg - 1, span);
    const out = new Array(nBasis);
    for (let j = 0; j < nBasis; j++) {
      const a = t[j + deg] - t[j];
      const b = t[j + deg + 1] - t[j + 1];
      out[j] = deg * ((a > 0 ? lower[j] / a : 0) - (b > 0 ? lower[j + 1] / b : 0));
    }
    return out;
  }

  // Basis row for one feature value. Outside the base interval the basis is extended linearly
  // (value and slope at the boundary), as sklearn does with extrapolation="linear".
  function splineBasis(x, t, deg) {
    const lo = t[deg];
    const hi = t[t.length - deg - 1];
    if (x >= lo && x <= hi) return basisInside(x, t, deg);
    const edge = x < lo ? lo : hi;
    const f = basisInside(edge, t, deg);
    const fp = basisDerivative(edge, t, deg);
    return f.map((v, i) => v + (x - edge) * fp[i]);
  }

  /* ---------- Model ---------- */

  function Model(artifact) {
    this.a = artifact;
    this.features = artifact.features;
    this.deg = artifact.spline.degree;
    this.nSplines = artifact.spline.n_splines;
    this.knots = artifact.spline.knots;
    this.mean = artifact.scaler.mean;
    this.scale = artifact.scaler.scale;
    this.coef = artifact.logistic.coef;
    this.intercept = artifact.logistic.intercept;
    if (this.coef.length !== this.features.length * this.nSplines) {
      throw new Error('Model artifact is inconsistent: coefficient count does not match features x splines.');
    }
  }

  // Centred log-odds contribution of feature j at value x.
  Model.prototype.contribution = function (j, x) {
    const b = splineBasis(x, this.knots[j], this.deg);
    let s = 0;
    for (let c = 0; c < this.nSplines; c++) {
      const col = j * this.nSplines + c;
      s += (this.coef[col] / this.scale[col]) * (b[c] - this.mean[col]);
    }
    return s;
  };

  Model.prototype.contributions = function (row) {
    return row.map((x, j) => this.contribution(j, x));
  };

  Model.prototype.logit = function (row) {
    let z = this.intercept;
    for (let j = 0; j < row.length; j++) z += this.contribution(j, row[j]);
    return z;
  };

  Model.prototype.predict = function (row) {
    return sigmoid(this.logit(row));
  };

  function sigmoid(z) {
    return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  }

  /* ---------- Input validation ---------- */

  // Returns {ok, values, errors[], warnings[]} for one row given as {featureName: rawValue}.
  function validateRow(model, raw) {
    const errors = [];
    const warnings = [];
    const values = [];
    model.features.forEach((f, j) => {
      const v = raw[f];
      if (v === undefined || v === null || String(v).trim() === '') {
        errors.push(f + ' is missing.');
        values.push(NaN);
        return;
      }
      const num = typeof v === 'number' ? v : Number(String(v).trim().replace(/,/g, '.'));
      if (!Number.isFinite(num)) {
        errors.push(f + ' must be a number (got "' + v + '").');
        values.push(NaN);
        return;
      }
      const unit = model.a.units ? model.a.units[j] : '';
      if (num < 0 && (unit === 'rpm' || unit === 'bar')) {
        errors.push(f + ' cannot be negative (got ' + num + ').');
        values.push(NaN);
        return;
      }
      values.push(num);
      const r = model.a.ranges[f];
      if (num < r.min || num > r.max) {
        warnings.push(f + ' = ' + num + ' is outside the range seen in training (' + r.min.toFixed(2) + ' to ' + r.max.toFixed(2) + '); the prediction extrapolates.');
      }
    });
    return {ok: errors.length === 0, values: values, errors: errors, warnings: warnings};
  }

  /* ---------- Decision tables, capacity and cost (all from out-of-fold development data) ---------- */

  function thresholdIndex(thresholds, t) {
    let best = 0;
    let bd = Infinity;
    thresholds.forEach((v, i) => {
      const d = Math.abs(v - t);
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  }

  function confusionAt(tab, i) {
    const tp = tab.tp[i], fp = tab.fp[i], fn = tab.fn[i], tn = tab.tn[i];
    const n = tp + fp + fn + tn;
    return {
      tp: tp, fp: fp, fn: fn, tn: tn, n: n,
      flaggedShare: (tp + fp) / n,
      precision: tp + fp > 0 ? tp / (tp + fp) : null,
      recall: tp + fn > 0 ? tp / (tp + fn) : null
    };
  }

  // Cost per engine in "inspection units": one unnecessary inspection = 1, one missed case = r.
  function costFor(c, r) {
    return (c.fp + r * c.fn) / c.n;
  }

  function strategies(oof, r, thresholdIdx) {
    const i = thresholdIdx;
    const model = confusionAt(oof, i);
    const n = model.n;
    const pos = model.tp + model.fn;
    const neg = model.fp + model.tn;
    return {
      model: {cost: costFor(model, r), conf: model},
      all: {cost: neg / n},
      none: {cost: (r * pos) / n}
    };
  }

  // Threshold with the lowest expected cost for a miss-cost ratio r.
  function bestThreshold(oof, r) {
    let best = 0;
    let bc = Infinity;
    oof.thresholds.forEach((t, i) => {
      const c = costFor(confusionAt(oof, i), r);
      if (c < bc) { bc = c; best = i; }
    });
    return {index: best, threshold: oof.thresholds[best], cost: bc};
  }

  // Share of true cases caught if only the top `pct` percent of engines (by risk) are inspected.
  function captureAt(capture, oof, pct) {
    const i = Math.min(Math.max(Math.round(pct), 1), 100) - 1;
    const n = capture.n[i];
    const tp = capture.tp[i];
    const pos = oof.prevalence * oof.n;
    return {n: n, tp: tp, precision: tp / n, recall: tp / pos, randomPrecision: oof.prevalence};
  }

  // Percentile of a score among development out-of-fold scores (0 to 100).
  function scorePercentile(quantiles, p) {
    let k = 0;
    while (k < quantiles.length && quantiles[k] <= p) k++;
    return Math.max(0, Math.min(100, ((k - 1) / (quantiles.length - 1)) * 100));
  }

  /* ---------- CSV ---------- */

  function detectDelimiter(headerLine) {
    const counts = {',': 0, ';': 0, '\t': 0};
    let inQ = false;
    for (const ch of headerLine) {
      if (ch === '"') inQ = !inQ;
      else if (!inQ && ch in counts) counts[ch]++;
    }
    return Object.keys(counts).sort((a, b) => counts[b] - counts[a])[0];
  }

  function parseCSV(text) {
    text = text.replace(/^﻿/, '');
    const firstLine = text.split(/\r?\n/, 1)[0] || '';
    const delim = detectDelimiter(firstLine);
    const rows = [];
    let row = [];
    let cur = '';
    let inQ = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (inQ) {
        if (ch === '"') {
          if (text[i + 1] === '"') { cur += '"'; i++; } else inQ = false;
        } else cur += ch;
      } else if (ch === '"') inQ = true;
      else if (ch === delim) { row.push(cur); cur = ''; }
      else if (ch === '\n' || ch === '\r') {
        if (ch === '\r' && text[i + 1] === '\n') i++;
        row.push(cur); cur = '';
        if (row.length > 1 || row[0] !== '') rows.push(row);
        row = [];
      } else cur += ch;
    }
    if (cur !== '' || row.length) {
      row.push(cur);
      if (row.length > 1 || row[0] !== '') rows.push(row);
    }
    return rows;
  }

  function normName(s) {
    return String(s).toLowerCase().replace(/[^a-z0-9]/g, '');
  }

  // Parse + validate + score a CSV. Returns {rows, invalid, missingColumns, extraColumns, error}.
  function scoreCSV(model, text) {
    const grid = parseCSV(text);
    if (grid.length < 2) return {error: 'The file needs a header row and at least one data row.', rows: [], invalid: []};
    const header = grid[0].map(function (h) { return h.trim(); });
    const index = {};
    const missingColumns = [];
    model.features.forEach(function (f) {
      const k = header.findIndex(function (h) { return normName(h) === normName(f); });
      if (k < 0) missingColumns.push(f); else index[f] = k;
    });
    const used = new Set(Object.values(index));
    // An optional engine identifier column is carried through to the results (it is never used by the model).
    const idIdx = header.findIndex(function (h, k) { return !used.has(k) && ID_COLUMN_NAMES.indexOf(normName(h)) >= 0; });
    if (idIdx >= 0) used.add(idIdx);
    const extraColumns = header.filter(function (_, k) { return !used.has(k); });
    if (missingColumns.length) {
      return {error: 'Missing required column(s): ' + missingColumns.join(', ') + '.', rows: [], invalid: [], missingColumns: missingColumns, extraColumns: extraColumns};
    }
    const rows = [];
    const invalid = [];
    for (let r = 1; r < grid.length; r++) {
      const raw = {};
      model.features.forEach(function (f) { raw[f] = grid[r][index[f]]; });
      const v = validateRow(model, raw);
      const rowNumber = r + 1; // spreadsheet row number: header is row 1
      if (!v.ok) { invalid.push({rowNumber: rowNumber, errors: v.errors}); continue; }
      const p = model.predict(v.values);
      rows.push({
        rowNumber: rowNumber, id: idIdx >= 0 ? String(grid[r][idIdx] === undefined ? '' : grid[r][idIdx]).trim() : '',
        values: v.values, risk: p, warnings: v.warnings,
        outOfRange: outsideTraining(model, v.values),
        driver: mainDriver(model, v.values)
      });
    }
    rows.sort(function (a, b) { return b.risk - a.risk || a.rowNumber - b.rowNumber; });
    rows.forEach(function (x, k) { x.rank = k + 1; });
    return {rows: rows, invalid: invalid, missingColumns: [], extraColumns: extraColumns, idColumn: idIdx >= 0 ? header[idIdx] : null};
  }

  const ID_COLUMN_NAMES = ['engineid', 'id', 'unitid', 'assetid', 'vehicleid', 'enginenumber', 'engineno', 'serialnumber', 'serial', 'equipmentid'];

  function csvEscape(v) {
    const s = String(v);
    return /[",\n\r;]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }

  // Recommended action for one fleet row. A row with readings outside the training range is never
  // labeled routine on the strength of a low score alone.
  function fleetAction(r, threshold, capacity) {
    const above = r.risk >= threshold;
    const within = capacity === null || r.rank <= capacity;
    if (above && within) return 'Inspect (human review)';
    if (above) return 'Above threshold, over capacity: schedule next';
    if (r.outOfRange && r.outOfRange.length) return 'Verify readings (outside training range) before routine monitoring';
    return 'Routine monitoring';
  }

  // Ranked inspection list as CSV text.
  function exportCSV(model, rows, threshold, capacity) {
    const head = ['rank', 'csv_row_number'].concat(model.features, [
      'risk_score', 'above_threshold', 'within_capacity', 'recommended_action',
      'main_model_driver', 'driver_odds_multiplier', 'review_focus', 'reliability_note'
    ]);
    const lines = [head.join(',')];
    rows.forEach(function (r) {
      const above = r.risk >= threshold;
      const within = capacity === null || r.rank <= capacity;
      const d = r.driver;
      const oor = r.outOfRange && r.outOfRange.length ? 'Outside training range: ' + r.outOfRange.join('; ') : '';
      lines.push([r.rank, r.rowNumber].concat(r.values, [
        r.risk.toFixed(6), above ? 'yes' : 'no', within ? 'yes' : 'no', fleetAction(r, threshold, capacity),
        d ? d.name : 'None (no single sensor driver)', d ? (d.oddsFactor > 1000 ? '>1000' : d.oddsFactor.toFixed(2)) : '', reviewFocus(d), oor
      ]).map(csvEscape).join(','));
    });
    return lines.join('\n') + '\n';
  }

  /* ---------- Fleet workflow: predict every engine first, prioritise inspections only when needed ---------- */

  const NEEDS_INSPECTION = 'Needs Inspection';
  const BELOW_THRESHOLD = 'Below Inspection Threshold';

  // Classification depends only on the model score and the threshold; never on inspection capacity.
  function recommendation(r, threshold) {
    return r.risk >= threshold ? NEEDS_INSPECTION : BELOW_THRESHOLD;
  }

  function fleetSummary(rows, threshold) {
    const total = rows.length;
    const needs = rows.filter(function (r) { return r.risk >= threshold; }).length;
    return {
      total: total, needs: needs, below: total - needs, pctFlagged: total ? needs / total : 0,
      outOfRange: rows.filter(function (r) { return r.outOfRange && r.outOfRange.length; }).length
    };
  }

  // Optional business step. Picks the highest-risk flagged engines up to the capacity. The scores are not touched.
  function prioritize(rows, threshold, capacity) {
    const flagged = rows.filter(function (r) { return r.risk >= threshold; })
      .sort(function (a, b) { return b.risk - a.risk || a.rowNumber - b.rowNumber; });
    const n = Number.isFinite(capacity) && capacity >= 1 ? Math.floor(capacity) : 0;
    const selected = flagged.slice(0, n);
    return {flagged: flagged.length, capacity: n, selected: selected, outside: flagged.length - selected.length, spare: Math.max(0, n - flagged.length)};
  }

  function detailCells(model, r, threshold) {
    const d = r.driver;
    const oor = r.outOfRange && r.outOfRange.length ? 'Outside training range: ' + r.outOfRange.join('; ') : '';
    return [r.risk.toFixed(6), recommendation(r, threshold),
      d ? d.name : 'None (no single sensor driver)', d ? (d.oddsFactor > 1000 ? '>1000' : d.oddsFactor.toFixed(2)) : '', reviewFocus(d), oor];
  }
  const DETAIL_HEAD = ['predicted_probability', 'inspection_recommendation', 'main_model_driver', 'driver_odds_multiplier', 'review_focus', 'reliability_note'];

  // Complete prediction results: every valid engine, in risk order, independent of inspection capacity.
  function exportAllCSV(model, rows, threshold) {
    const lines = [['risk_rank', 'engine_id', 'csv_row_number'].concat(model.features, DETAIL_HEAD).join(',')];
    rows.forEach(function (r) {
      lines.push([r.rank, r.id || '', r.rowNumber].concat(r.values, detailCells(model, r, threshold)).map(csvEscape).join(','));
    });
    return lines.join('\n') + '\n';
  }

  // Prioritised inspection list: the selected highest-risk flagged engines only.
  function exportPriorityCSV(model, rows, threshold, capacity) {
    const p = prioritize(rows, threshold, capacity);
    const lines = [['inspection_priority', 'risk_rank', 'engine_id', 'csv_row_number'].concat(model.features, DETAIL_HEAD).join(',')];
    p.selected.forEach(function (r, k) {
      lines.push([k + 1, r.rank, r.id || '', r.rowNumber].concat(r.values, detailCells(model, r, threshold)).map(csvEscape).join(','));
    });
    return lines.join('\n') + '\n';
  }

  /* ---------- Explanations and recommendations (derived from the fitted model's contributions) ---------- */

  // A sensor "meaningfully" moves the score when its log-odds contribution is at least this large,
  // i.e. it multiplies the odds of the Needs Maintenance label by about 1.1 or more (or 1/1.1 or less).
  const MEANINGFUL = 0.10;

  function unitLabel(u) {
    return u === 'degC' ? '°C' : u === 'rpm' ? 'RPM' : (u || '');
  }

  function fmtNum(v) {
    const a = Math.abs(v);
    return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
  }

  function fmtOdds(f) {
    return f > 1000 ? 'more than 1,000' : f < 0.001 ? 'less than 0.001' : f.toFixed(2);
  }

  function withUnit(v, u) {
    const l = unitLabel(u);
    return fmtNum(v) + (l ? ' ' + l : '');
  }

  function outsideTraining(model, values) {
    const out = [];
    model.features.forEach(function (f, j) {
      const r = model.a.ranges[f];
      if (values[j] < r.min || values[j] > r.max) out.push(f);
    });
    return out;
  }

  // Largest meaningful positive contribution, or null when no sensor raises the score meaningfully.
  function mainDriver(model, values) {
    const c = model.contributions(values);
    let best = -1;
    c.forEach(function (v, j) { if (v >= MEANINGFUL && (best < 0 || v > c[best])) best = j; });
    return best < 0 ? null : {index: best, name: model.features[best], contribution: c[best], oddsFactor: Math.exp(c[best])};
  }

  function reviewFocus(driver) {
    if (!driver) return 'No single sensor driver: review per routine';
    const topic = {
      'Engine RPM': 'RPM and the operating conditions it was recorded under',
      'Lub Oil Pressure': 'lubricating-oil pressure reading',
      'Fuel Pressure': 'fuel-pressure reading',
      'Coolant Pressure': 'coolant-pressure reading',
      'Lub Oil Temp': 'lubricating-oil temperature reading',
      'Coolant Temp': 'coolant-temperature reading'
    }[driver.name] || (driver.name + ' reading');
    return 'Verify ' + topic;
  }

  const AREA_TEXT = {
    'Engine RPM': 'Verify the RPM reading and the operating conditions it was recorded under (for example idling versus working load).',
    'Lub Oil Pressure': 'Recheck the lubricating-oil pressure reading and compare it with the manufacturer’s specification for this engine and operating condition.',
    'Fuel Pressure': 'Verify the fuel-pressure measurement and review the fuel-delivery readings for this engine against the manufacturer’s specification.',
    'Coolant Pressure': 'Recheck the coolant-pressure reading and review the cooling-system readings against the manufacturer’s specification.',
    'Lub Oil Temp': 'Verify the lubricating-oil temperature reading and the recent operating conditions, and compare it with the manufacturer’s specification.',
    'Coolant Temp': 'Verify the coolant-temperature reading and compare it with the engine’s normal operating range.'
  };

  function pctText(x, d) {
    return (x * 100).toFixed(d === undefined ? 0 : d) + '%';
  }

  function relationText(it) {
    const rng = fmtNum(it.p25) + ' to ' + fmtNum(it.p75) + (unitLabel(it.unit) ? ' ' + unitLabel(it.unit) : '');
    const med = 'median ' + withUnit(it.median, it.unit);
    if (it.outsideTraining) return 'outside the range seen in training (' + fmtNum(it.trainMin) + ' to ' + fmtNum(it.trainMax) + ')';
    if (it.relation === 'above') return 'above the typical range for development engines (' + rng + '; ' + med + ')';
    if (it.relation === 'below') return 'below the typical range for development engines (' + rng + '; ' + med + ')';
    return 'within the typical range for development engines (' + rng + '; ' + med + ')';
  }

  // hi: at or above the threshold; mid: within 0.10 below it; lo: otherwise.
  function levelFor(p, threshold) {
    return p >= threshold ? 'hi' : p >= threshold - 0.10 ? 'mid' : 'lo';
  }

  const TOPIC = {
    'Engine RPM': 'RPM reading',
    'Lub Oil Pressure': 'lubricating-oil pressure reading',
    'Fuel Pressure': 'fuel-pressure reading',
    'Coolant Pressure': 'coolant-pressure reading',
    'Lub Oil Temp': 'lubricating-oil temperature reading',
    'Coolant Temp': 'coolant-temperature reading'
  };

  // Three demo engines chosen from real rows (held-out test rows stored with the model), classified by the
  // trained model itself at the given threshold. Only in-range rows are used. Targets are relative to the
  // threshold: clearly above (+0.30), just below (-0.05), clearly below (-0.25).
  function pickExamples(model, rows, threshold) {
    const targets = {hi: threshold + 0.30, mid: threshold - 0.05, lo: threshold - 0.25};
    const best = {hi: null, mid: null, lo: null};
    rows.forEach(function (row, i) {
      if (outsideTraining(model, row).length) return;
      const p = model.predict(row);
      const level = levelFor(p, threshold);
      const d = Math.abs(p - targets[level]);
      if (!best[level] || d < best[level].d) best[level] = {index: i, p: p, level: level, d: d};
    });
    return best;
  }

  // Everything the Predict page shows about one engine, computed from the fitted model.
  function explainPrediction(model, values, threshold) {
    const F = model.features;
    const U = model.a.units || [];
    const contribs = model.contributions(values);
    const logit = model.intercept + contribs.reduce(function (s, v) { return s + v; }, 0);
    const p = sigmoid(logit);
    // Reference probability = logistic(intercept): the model's prediction when every sensor effect is at its
    // development-data average (centred contribution 0). It is NOT the observed share of labelled engines.
    const refP = sigmoid(model.intercept);
    const prev = model.a.metrics && typeof model.a.metrics.dev_prevalence === 'number' ? model.a.metrics.dev_prevalence : null;
    const referenceNote = 'The model’s reference probability is ' + pctText(refP, 0) + '. It is what the model predicts when every sensor’s effect sits at its average in the development data, so it is the starting point to which the sensor effects are added. It is a modeling reference, not the observed share of development engines that carry the label' + (prev === null ? '' : ' (' + pctText(prev, 0) + ')') + ', and not the risk of an actual "average" engine.';
    const totalAbs = contribs.reduce(function (s, v) { return s + Math.abs(v); }, 0);

    const items = F.map(function (f, j) {
      const r = model.a.ranges[f];
      const c = contribs[j];
      const v = values[j];
      const it = {
        index: j, name: f, unit: U[j] || '', value: v, median: r.median, p25: r.p25, p75: r.p75,
        trainMin: r.min, trainMax: r.max,
        contribution: c, oddsFactor: Math.exp(c), share: totalAbs > 0 ? Math.abs(c) / totalAbs : 0,
        direction: c >= MEANINGFUL ? 'up' : c <= -MEANINGFUL ? 'down' : 'neutral',
        relation: v < r.p25 ? 'below' : v > r.p75 ? 'above' : 'within',
        outsideTraining: v < r.min || v > r.max
      };
      const factor = fmtOdds(it.oddsFactor);
      if (it.direction === 'up') {
        it.sentence = f + ' is ' + withUnit(v, it.unit) + ', ' + relationText(it) + '. In the trained model this raises the odds of the Needs Maintenance label (they are multiplied by ' + factor + ').';
      } else if (it.direction === 'down') {
        it.sentence = f + ' is ' + withUnit(v, it.unit) + ', ' + relationText(it) + '. In the trained model this lowers the odds of the Needs Maintenance label (they are multiplied by ' + factor + ').';
      } else {
        it.sentence = f + ' is ' + withUnit(v, it.unit) + ', ' + relationText(it) + '. Its effect on the score is small (odds multiplied by ' + factor + ').';
      }
      return it;
    });

    const byMag = function (a, b) { return Math.abs(b.contribution) - Math.abs(a.contribution); };
    const up = items.filter(function (i) { return i.direction === 'up'; }).sort(byMag).slice(0, 3);
    const down = items.filter(function (i) { return i.direction === 'down'; }).sort(byMag).slice(0, 3);
    const outOfRange = items.filter(function (i) { return i.outsideTraining; });
    const reliable = outOfRange.length === 0;

    const above = p >= threshold;
    const near = !above && p >= threshold - 0.10;
    const level = levelFor(p, threshold);
    const oorNames = outOfRange.map(function (i) { return i.name; });

    let priority;
    if (above) priority = reliable ? 'Inspection recommended — above selected threshold' : 'Inspection recommended — above selected threshold (verify readings first)';
    else if (!reliable) priority = 'Verify readings before routine monitoring';
    else if (near) priority = 'Watch list';
    else priority = 'Routine monitoring';

    // --- plain-English narrative ---
    const lines = [];
    const rel = above ? (p === threshold ? 'at' : 'above') : 'below';
    lines.push('The model estimates a ' + pctText(p, 1) + ' probability of the Needs Maintenance label for this engine. That is ' + rel + ' your ' + pctText(threshold, 0) + ' inspection threshold, so ' +
      (above ? 'the engine is flagged for inspection.' : near ? 'it is not flagged, but it is close enough to watch.' : 'it is not flagged.'));
    if (up.length) {
      lines.push('Measured against each sensor’s average effect in the development data, ' + (up.length === 1 ? 'one reading raises the score meaningfully: ' : 'these readings raise the score the most: ') + up.map(function (i) { return i.name; }).join(', ') + '. These are sensor-by-sensor patterns the model learned from this dataset, not proven physical causes.');
    } else {
      lines.push('No single sensor raises this engine’s score meaningfully (each effect is smaller than a ' + MEANINGFUL.toFixed(2) + ' log-odds change, about 10% on the odds). The model’s reference probability is ' + pctText(refP, 0) + ' (its starting point when every sensor effect is average' + (prev === null ? '' : '; the observed share of development engines with the label is ' + pctText(prev, 0)) + '), and ' + (down.length ? 'the readings listed below move the score down from there.' : 'the readings leave the score close to that reference.') +
        (above && refP >= threshold ? ' The reference level alone is above your current threshold, so the threshold you choose matters more here than any individual reading.' : ''));
    }
    if (down.length) lines.push('These readings pull the score down: ' + down.map(function (i) { return i.name; }).join(', ') + '.');
    if (!reliable) lines.push('Reliability warning: ' + oorNames.join(', ') + (oorNames.length === 1 ? ' is' : ' are') + ' outside the range seen in training, so the model is extrapolating and the score may not be dependable.');

    // --- recommendations (areas to investigate, actions, caution) ---
    const areas = up.slice(0, 3).map(function (i) {
      return {feature: i.name, text: AREA_TEXT[i.name] || ('Recheck the ' + i.name + ' reading and compare it with the manufacturer’s specification.')};
    });
    const caution = reliable ? null :
      'Verify the ' + oorNames.join(', ') + ' reading' + (oorNames.length === 1 ? '' : 's') + ' first. The value is outside the training data, so the model is extrapolating. Request human review before relying on this score' + (above ? '' : ', and do not treat a low score as a reason for routine monitoring') + '.';
    let actions;
    if (above) actions = ['Add this engine to the inspection queue. If inspection capacity is limited, use Fleet Priorities to rank it against other engines by risk score.', 'Request an inspection by a qualified technician. The model score is a ranking aid, not a diagnosis.', 'Record the score, threshold and readings used for this decision.'];
    else if (near) actions = ['Keep the engine on a watch list and re-check the readings at the next opportunity.', 'If inspection capacity is available, include it with the higher-risk engines.'];
    else if (reliable) actions = ['Continue routine monitoring and normal safety procedures.', 'A low score is not a guarantee. Keep acting on alarms, operator reports and scheduled inspections.'];
    else actions = ['Keep normal safety procedures in place until the readings have been verified.'];

    const steps = [];
    if (caution) steps.push({kind: 'caution', text: caution});
    areas.forEach(function (a) { steps.push({kind: 'area', text: a.text}); });
    if (up.length || above || near) steps.push({kind: 'area', text: 'Compare the readings with the manufacturer’s specifications, which are not part of this dataset.'});
    actions.forEach(function (a) { steps.push({kind: 'action', text: a}); });

    // --- two-line summary for the "at a glance" card ---
    const relShort = function (i) { return i.outsideTraining ? 'outside the training range' : i.relation === 'above' ? 'above the typical range' : i.relation === 'below' ? 'below the typical range' : 'within the typical range'; };
    const topic = function (n) { return TOPIC[n] || (n + ' reading'); };
    const top2 = up.slice(0, 2);
    let glanceReason;
    if (!reliable) glanceReason = oorNames.join(', ') + (oorNames.length === 1 ? ' is' : ' are') + ' outside the range seen in training, so the score may not be dependable.';
    else if (up.length) glanceReason = (level === 'hi' ? 'Main reason: ' : 'Closest concern: ') + top2.map(function (i) { return i.name + ' (' + withUnit(i.value, i.unit) + ', ' + relShort(i) + ')'; }).join(' and ') + (top2.length === 1 ? ' raises' : ' raise') + ' the score most.';
    else if (down.length) glanceReason = 'No single sensor raises the score. ' + down.slice(0, 2).map(function (i) { return i.name; }).join(' and ') + (down.slice(0, 2).length === 1 ? ' pulls' : ' pull') + ' it down from the model’s reference level (' + pctText(refP, 0) + ').';
    else glanceReason = 'No single sensor stands out. The score stays close to the model’s reference level (' + pctText(refP, 0) + ').';
    let glanceNext;
    if (!reliable) glanceNext = 'Verify the ' + oorNames.join(', ') + ' reading' + (oorNames.length === 1 ? '' : 's') + ' first, then request human review before relying on this score.';
    else if (above) glanceNext = 'Add it to the inspection queue and ask a qualified technician to review it.' + (up.length ? ' Start by checking the ' + top2.map(function (i) { return topic(i.name); }).join(' and ') + '.' : '');
    else if (near) glanceNext = 'Put it on the watch list and re-check the readings soon.';
    else glanceNext = 'No extra inspection suggested. Keep routine monitoring and normal safety procedures.';

    return {
      glance: {headline: priority, reason: glanceReason, next: glanceNext},
      p: p, logit: logit, referenceProbability: refP, observedPrevalence: prev, referenceNote: referenceNote, level: level, priority: priority, threshold: threshold,
      items: items, up: up, down: down, outOfRange: outOfRange, reliable: reliable,
      narrative: lines, areas: areas, actions: actions, caution: caution, steps: steps
    };
  }

  const api = {
    splineBasis: splineBasis, Model: Model, sigmoid: sigmoid, validateRow: validateRow,
    thresholdIndex: thresholdIndex, confusionAt: confusionAt, costFor: costFor, strategies: strategies,
    bestThreshold: bestThreshold, captureAt: captureAt, scorePercentile: scorePercentile,
    parseCSV: parseCSV, scoreCSV: scoreCSV, exportCSV: exportCSV, fleetAction: fleetAction,
    explainPrediction: explainPrediction, pickExamples: pickExamples, levelFor: levelFor, mainDriver: mainDriver, reviewFocus: reviewFocus,
    outsideTraining: outsideTraining, MEANINGFUL: MEANINGFUL, withUnit: withUnit,
    recommendation: recommendation, fleetSummary: fleetSummary, prioritize: prioritize,
    exportAllCSV: exportAllCSV, exportPriorityCSV: exportPriorityCSV,
    NEEDS_INSPECTION: NEEDS_INSPECTION, BELOW_THRESHOLD: BELOW_THRESHOLD
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SmartOpsCore = api;
})(typeof window !== 'undefined' ? window : globalThis);
