/* SmartOps core: model inference, decision/cost maths and CSV handling.
 * No dependencies. Works in the browser (window.SmartOpsCore) and in Node (module.exports).
 *
 * The model is a spline logistic regression exported from scikit-learn (see
 * notebooks/dashboard_model_export.ipynb). Inference re-implements, exactly:
 *   SplineTransformer(degree=3, knots="quantile", extrapolation="linear")
 *   -> StandardScaler -> LogisticRegression
 * and is tested against scikit-learn's own predictions (dashboard/tests).
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
        rowNumber: rowNumber, values: v.values, risk: p, warnings: v.warnings,
        outOfRange: outsideTraining(model, v.values),
        driver: mainDriver(model, v.values)
      });
    }
    rows.sort(function (a, b) { return b.risk - a.risk || a.rowNumber - b.rowNumber; });
    rows.forEach(function (x, k) { x.rank = k + 1; });
    return {rows: rows, invalid: invalid, missingColumns: [], extraColumns: extraColumns};
  }

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
        d ? d.name : 'None (baseline risk)', d ? (d.oddsFactor > 1000 ? '>1000' : d.oddsFactor.toFixed(2)) : '', reviewFocus(d), oor
      ]).map(csvEscape).join(','));
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

  // Everything the Predict page shows about one engine, computed from the fitted model.
  function explainPrediction(model, values, threshold) {
    const F = model.features;
    const U = model.a.units || [];
    const contribs = model.contributions(values);
    const logit = model.intercept + contribs.reduce(function (s, v) { return s + v; }, 0);
    const p = sigmoid(logit);
    const baseP = sigmoid(model.intercept);
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
    const level = above ? 'hi' : near ? 'mid' : 'lo';
    const oorNames = outOfRange.map(function (i) { return i.name; });

    let priority;
    if (above) priority = reliable ? 'High priority: inspect' : 'High priority: inspect (verify readings first)';
    else if (!reliable) priority = 'Verify readings before routine monitoring';
    else if (near) priority = 'Watch list';
    else priority = 'Routine monitoring';

    // --- plain-English narrative ---
    const lines = [];
    const rel = above ? (p === threshold ? 'at' : 'above') : 'below';
    lines.push('The model estimates a ' + pctText(p, 1) + ' probability of the Needs Maintenance label for this engine. That is ' + rel + ' your ' + pctText(threshold, 0) + ' inspection threshold, so ' +
      (above ? 'the engine is flagged for inspection.' : near ? 'it is not flagged, but it is close enough to watch.' : 'it is not flagged.'));
    if (up.length) {
      lines.push('Compared with the average development engine, ' + (up.length === 1 ? 'one reading raises the score meaningfully: ' : 'these readings raise the score the most: ') + up.map(function (i) { return i.name; }).join(', ') + '. These are sensor-by-sensor patterns the model learned from this dataset, not proven physical causes.');
    } else {
      lines.push('No single sensor raises this engine’s score meaningfully (each effect is smaller than a ' + MEANINGFUL.toFixed(2) + ' log-odds change, about 10% on the odds). The model starts from a baseline risk of ' + pctText(baseP, 0) + ' (the Needs Maintenance label is common in the training data) and ' + (down.length ? 'the readings listed below move the score down from there.' : 'the readings leave the score close to that baseline.') +
        (above && baseP >= threshold ? ' The baseline alone is above your current threshold, so the threshold you choose matters more here than any individual reading.' : ''));
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
    if (above) actions = ['Prioritize this engine in the inspection queue.', 'Request an inspection by a qualified technician. The model score is a ranking aid, not a diagnosis.', 'Record the score, threshold and readings used for this decision.'];
    else if (near) actions = ['Keep the engine on a watch list and re-check the readings at the next opportunity.', 'If inspection capacity is available, include it with the higher-risk engines.'];
    else if (reliable) actions = ['Continue routine monitoring and normal safety procedures.', 'A low score is not a guarantee. Keep acting on alarms, operator reports and scheduled inspections.'];
    else actions = ['Keep normal safety procedures in place until the readings have been verified.'];

    const steps = [];
    if (caution) steps.push({kind: 'caution', text: caution});
    areas.forEach(function (a) { steps.push({kind: 'area', text: a.text}); });
    if (up.length || above || near) steps.push({kind: 'area', text: 'Compare the readings with the manufacturer’s specifications, which are not part of this dataset.'});
    actions.forEach(function (a) { steps.push({kind: 'action', text: a}); });

    return {
      p: p, logit: logit, baseline: baseP, level: level, priority: priority, threshold: threshold,
      items: items, up: up, down: down, outOfRange: outOfRange, reliable: reliable,
      narrative: lines, areas: areas, actions: actions, caution: caution, steps: steps
    };
  }

  const api = {
    splineBasis: splineBasis, Model: Model, sigmoid: sigmoid, validateRow: validateRow,
    thresholdIndex: thresholdIndex, confusionAt: confusionAt, costFor: costFor, strategies: strategies,
    bestThreshold: bestThreshold, captureAt: captureAt, scorePercentile: scorePercentile,
    parseCSV: parseCSV, scoreCSV: scoreCSV, exportCSV: exportCSV, fleetAction: fleetAction,
    explainPrediction: explainPrediction, mainDriver: mainDriver, reviewFocus: reviewFocus,
    outsideTraining: outsideTraining, MEANINGFUL: MEANINGFUL, withUnit: withUnit
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SmartOpsCore = api;
})(typeof window !== 'undefined' ? window : globalThis);
