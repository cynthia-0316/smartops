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
      rows.push({rowNumber: rowNumber, values: v.values, risk: p, warnings: v.warnings});
    }
    rows.sort(function (a, b) { return b.risk - a.risk || a.rowNumber - b.rowNumber; });
    rows.forEach(function (x, k) { x.rank = k + 1; });
    return {rows: rows, invalid: invalid, missingColumns: [], extraColumns: extraColumns};
  }

  function csvEscape(v) {
    const s = String(v);
    return /[",\n\r;]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }

  // Ranked inspection list as CSV text. `flagged(row)` decides the recommendation.
  function exportCSV(model, rows, threshold, capacity) {
    const head = ['rank', 'csv_row_number'].concat(model.features, ['risk_score', 'above_threshold', 'within_capacity', 'recommended_action']);
    const lines = [head.join(',')];
    rows.forEach(function (r) {
      const above = r.risk >= threshold;
      const within = capacity === null || r.rank <= capacity;
      const action = above && within ? 'Inspect (human review)' : above ? 'Above threshold, over capacity: schedule next' : 'Routine monitoring';
      lines.push([r.rank, r.rowNumber].concat(r.values, [r.risk.toFixed(6), above ? 'yes' : 'no', within ? 'yes' : 'no', action]).map(csvEscape).join(','));
    });
    return lines.join('\n') + '\n';
  }

  const api = {
    splineBasis: splineBasis, Model: Model, sigmoid: sigmoid, validateRow: validateRow,
    thresholdIndex: thresholdIndex, confusionAt: confusionAt, costFor: costFor, strategies: strategies,
    bestThreshold: bestThreshold, captureAt: captureAt, scorePercentile: scorePercentile,
    parseCSV: parseCSV, scoreCSV: scoreCSV, exportCSV: exportCSV
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SmartOpsCore = api;
})(typeof window !== 'undefined' ? window : globalThis);
