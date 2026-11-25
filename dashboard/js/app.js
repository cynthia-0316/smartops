/* SmartOps dashboard UI. All numbers come from the exported model artifact (window.SMARTOPS_MODEL)
 * and from SmartOpsCore; nothing here is hardcoded or simulated. */
(function () {
  'use strict';
  const C = window.SmartOpsCore;
  const art = window.SMARTOPS_MODEL;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const pct = (x, d = 0) => (x * 100).toFixed(d) + '%';
  const f2 = (x) => Number(x).toFixed(2);
  const f4 = (x) => Number(x).toFixed(4);

  /* ---------- navigation ---------- */
  const TABS = ['overview', 'predict', 'fleet', 'insights', 'business', 'responsible'];
  function route() {
    const id = TABS.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'overview';
    TABS.forEach((t) => { $(t).hidden = t !== id; });
    document.querySelectorAll('#nav a').forEach((a) => {
      if (a.dataset.tab === id) { a.setAttribute('aria-current', 'page'); a.scrollIntoView({inline: 'center', block: 'nearest'}); } else a.removeAttribute('aria-current');
    });
    window.scrollTo(0, 0);
    document.title = 'SmartOps · ' + id.charAt(0).toUpperCase() + id.slice(1);
  }
  window.addEventListener('hashchange', route);
  route();

  if (!art) {
    const b = $('banner');
    b.className = 'banner err'; b.hidden = false;
    b.textContent = 'Model file not found. Run notebooks/dashboard_model_export.ipynb in Google Colab and copy model_data.js and model.json into dashboard/model/ (see dashboard/README.md).';
    $('ov-cards').innerHTML = '<div class="card"><div class="k">Model</div><div class="v">Not loaded</div><div class="s">No predictions are available until the trained model is exported.</div></div>';
    document.querySelectorAll('#form button[type=submit], #csv, #btn-sample, #btn-sample-load, #btn-reset').forEach((e) => { e.disabled = true; });
    return;
  }

  const model = new C.Model(art);
  const F = art.features;
  const U = art.units;
  const M = art.metrics;
  const oof = art.oof;
  const DEFAULTS = {threshold: 0.5, ratio: 1, unit: 100, capPct: 10};
  const RESULT_PLACEHOLDER = '<div class="panel"><h2>Result</h2><p class="muted">Pick a one-click example or enter readings, then press <strong>Analyze Engine</strong>. You will get the risk, the main reason, and what to check first.</p></div>';
  const state = {threshold: DEFAULTS.threshold, ratio: DEFAULTS.ratio, unit: DEFAULTS.unit, capPct: DEFAULTS.capPct, fleet: null, capN: null, capErr: false, fleetLimit: 100, openRows: new Set(), lastRow: null, detailsOpen: false};

  if (art.meta.fixture) {
    const b = $('banner');
    b.className = 'banner'; b.hidden = false;
    b.textContent = 'DEVELOPMENT FIXTURE: this model file is only for testing the interface and is not the trained SmartOps model. Do not use these numbers.';
  }

  /* ---------- shared settings ---------- */
  function setThreshold(v, from) {
    state.threshold = Math.min(0.95, Math.max(0.05, Math.round(v * 100) / 100));
    ['thr-fleet', 'thr-biz', 'thr-pred'].forEach((id) => { const e = $(id); if (e && id !== from) e.value = state.threshold; });
    ['thr-fleet-out', 'thr-biz-out', 'thr-pred-out'].forEach((id) => { const e = $(id); if (e) e.textContent = f2(state.threshold); });
    renderOverview(); renderBusiness(); drawCurve(); if (state.fleet) renderFleet(); if (state.lastRow) renderResult();
  }
  const thrIdx = () => C.thresholdIndex(oof.thresholds, state.threshold);

  /* ---------- SVG helpers ---------- */
  function lineChart(el, o) {
    const W = 640, H = 300, L = 56, R = 16, T = 14, B = 46;
    const xs = o.series.flatMap((s) => s.pts.map((p) => p[0]));
    const ys = o.series.flatMap((s) => s.pts.map((p) => p[1]));
    const x0 = o.xmin ?? Math.min(...xs), x1 = o.xmax ?? Math.max(...xs);
    const y0 = o.ymin ?? Math.min(...ys), y1 = o.ymax ?? Math.max(...ys);
    const sx = (x) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R);
    const sy = (y) => H - B - ((y - y0) / (y1 - y0 || 1)) * (H - T - B);
    let g = '';
    for (let i = 0; i <= 4; i++) {
      const yv = y0 + ((y1 - y0) * i) / 4, xv = x0 + ((x1 - x0) * i) / 4;
      g += `<line x1="${L}" x2="${W - R}" y1="${sy(yv)}" y2="${sy(yv)}" stroke="#e5eaf3"/>`;
      g += `<text x="${L - 8}" y="${sy(yv) + 4}" text-anchor="end">${o.yFmt ? o.yFmt(yv) : yv.toFixed(2)}</text>`;
      g += `<text x="${sx(xv)}" y="${H - B + 18}" text-anchor="middle">${o.xFmt ? o.xFmt(xv) : xv.toFixed(2)}</text>`;
    }
    (o.vlines || []).forEach((v) => {
      g += `<line x1="${sx(v.x)}" x2="${sx(v.x)}" y1="${T}" y2="${H - B}" stroke="${v.color || '#9aa6bd'}" stroke-dasharray="${v.dash || '3 4'}"/>`;
      if (v.label) g += `<text x="${sx(v.x) + 4}" y="${T + 10}" style="fill:${v.color || '#51607a'}">${esc(v.label)}</text>`;
    });
    (o.hlines || []).forEach((h) => {
      g += `<line x1="${L}" x2="${W - R}" y1="${sy(h.y)}" y2="${sy(h.y)}" stroke="${h.color || '#9aa6bd'}" stroke-dasharray="${h.dash || '5 4'}"/>`;
      if (h.label) g += `<text x="${W - R - 4}" y="${sy(h.y) - 5}" text-anchor="end" style="fill:${h.color || '#51607a'}">${esc(h.label)}</text>`;
    });
    o.series.forEach((s) => {
      const d = s.pts.map((p, i) => (i ? 'L' : 'M') + sx(p[0]).toFixed(1) + ' ' + sy(p[1]).toFixed(1)).join('');
      g += `<path d="${d}" fill="none" stroke="${s.color || '#1f4fd8'}" stroke-width="${s.w || 2.5}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''} stroke-linejoin="round"/>`;
    });
    (o.points || []).forEach((p) => {
      g += `<circle cx="${sx(p.x)}" cy="${sy(p.y)}" r="${p.r || 5}" fill="${p.color || '#1f4fd8'}" stroke="#fff" stroke-width="2"/>`;
      if (p.label) g += `<text x="${sx(p.x) + 8}" y="${sy(p.y) - 8}" class="ttl">${esc(p.label)}</text>`;
    });
    g += `<text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">${esc(o.xLabel || '')}</text>`;
    g += `<text transform="translate(14 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${esc(o.yLabel || '')}</text>`;
    const legend = o.series.filter((s) => s.label).map((s) => `<span style="color:${s.color || '#1f4fd8'}">&#9632;</span> ${esc(s.label)}`).join(' &nbsp; ');
    el.innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria)}">${g}</svg>` + (legend ? `<p class="hint">${legend}</p>` : '');
  }

  function barChart(el, items, o) {
    const W = 480, rowH = 36, L = 128, R = 56, T = 8;
    const H = T + items.length * rowH + 8;
    const diverging = o.diverging;
    const mags = items.map((i) => Math.abs(i.v)).sort((a, b) => b - a);
    // One extreme value (for example an extrapolated, out-of-range sensor) must not squash the other bars:
    // shorten that bar and say so. The printed number is always the true value.
    const cap = diverging && mags.length > 1 && mags[0] > 4 * Math.max(mags[1], 0.25) ? 4 * Math.max(mags[1], 0.25) : null;
    const mx = (cap || mags[0]) || 1;
    const half = (W - L - R) / 2;
    const zero = diverging ? L + half : L;
    const scale = (diverging ? half : W - L - R) / mx;
    let g = diverging ? `<line x1="${zero}" x2="${zero}" y1="${T}" y2="${H - 8}" stroke="#9aa6bd"/>` : '';
    let clipped = false;
    items.forEach((it, i) => {
      const y = T + i * rowH;
      const isClipped = cap !== null && Math.abs(it.v) > cap;
      if (isClipped) clipped = true;
      const w = Math.min(Math.abs(it.v), cap || Infinity) * scale;
      const x = it.v >= 0 ? zero : zero - w;
      const color = diverging ? (it.v >= 0 ? '#c2410c' : '#1f4fd8') : '#1f4fd8';
      g += `<text x="${L - 10}" y="${y + 17}" text-anchor="end" class="ttl">${esc(it.label)}</text>`;
      g += `<rect x="${x}" y="${y + 6}" width="${Math.max(w, 1)}" height="20" rx="4" fill="${color}"/>`;
      if (isClipped) { // break marks at the shortened end
        const bx = it.v >= 0 ? x + w - 10 : x + 10;
        g += `<path d="M${bx - 3} ${y + 26} l6 -20 M${bx + 3} ${y + 26} l6 -20" stroke="#fff" stroke-width="2.5" fill="none"/>`;
      }
      const text = o.fmt(it.v);
      const roomOutside = it.v >= 0 ? W - (zero + w + 6) : zero - w - 6 - L;
      const needed = text.length * 6.6 + 4;
      let tx, anchor, fill = '';
      if (roomOutside >= needed) { tx = it.v >= 0 ? zero + w + 6 : zero - w - 6; anchor = it.v >= 0 ? 'start' : 'end'; }
      else { const pad = isClipped ? 26 : 6; tx = it.v >= 0 ? zero + w - pad : zero - w + pad; anchor = it.v >= 0 ? 'end' : 'start'; fill = ' style="fill:#fff;font-weight:600"'; }
      g += `<text x="${tx}" y="${y + 21}" text-anchor="${anchor}"${fill}>${esc(text)}</text>`;
      if (it.sub) g += `<text x="${L - 10}" y="${y + 31}" text-anchor="end" style="font-size:10.5px">${esc(it.sub)}</text>`;
    });
    el.innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria)}">${g}</svg>` +
      (clipped ? '<p class="hint">One bar is drawn shortened so the other effects stay readable. The number printed on it is the true value. Very large effects usually mean a reading outside the training range.</p>' : '');
  }

  /* ---------- Overview ---------- */
  function crossoverRatio() {
    for (let r = 0.5; r <= 10.001; r += 0.5) {
      const b = C.bestThreshold(oof, r);
      const all = C.strategies(oof, r, 0).all.cost;
      if (b.cost >= all - 1e-4) return r;
    }
    return null;
  }
  function renderOverview() {
    const c = C.confusionAt(oof, thrIdx());
    const t = M.test;
    const cards = [
      {k: 'Model', v: 'Penalised logistic GAM', s: 'One additive model, one smooth curve per sensor, fitted on development data only.', cls: 'accent', small: true},
      {k: 'Validated performance (held-out test)', v: 'AUC ' + f2(t.auc), s: `95% CI ${f2(t.auc_ci[0])} to ${f2(t.auc_ci[1])} on ${t.n.toLocaleString()} engines. Cross-validation ${f2(M.cv.auc_mean)} ± ${f2(M.cv.auc_std)}.`},
      {k: `Engines flagged at threshold ${f2(state.threshold)}`, v: pct(c.flaggedShare), s: `Development data, out-of-fold: ${Math.round(c.flaggedShare * c.n).toLocaleString()} of ${c.n.toLocaleString()} engines.`},
      {k: 'Engines above the threshold in your upload', v: state.fleet ? state.fleet.rows.filter((r) => r.risk >= state.threshold).length.toLocaleString() : 'None yet', s: state.fleet ? `of ${state.fleet.rows.length.toLocaleString()} valid rows above the threshold.` : 'Upload a fleet file on Fleet Priorities.'}
    ];
    $('ov-cards').innerHTML = cards.map((x) => `<div class="card ${x.cls || ''}"><div class="k">${esc(x.k)}</div><div class="v" ${x.small ? 'style="font-size:21px"' : ''}>${esc(x.v)}</div><div class="s">${esc(x.s)}</div></div>`).join('');

    const prev = oof.prevalence;
    const b1 = C.bestThreshold(oof, 1);
    const s1 = C.strategies(oof, 1, b1.index);
    const saving = 1 - s1.model.cost / Math.min(s1.all.cost, s1.none.cost);
    const rx = crossoverRatio();
    $('ov-reco-body').innerHTML =
      `<p>At threshold <strong>${f2(state.threshold)}</strong> the model flags <strong>${pct(c.flaggedShare)}</strong> of development engines. <strong>${pct(c.precision)}</strong> of the flagged engines carry the <em>Needs Maintenance</em> label (${pct(prev)} by chance) and <strong>${pct(c.recall)}</strong> of all labelled cases are found.</p>` +
      `<div class="next"><strong>Recommended use.</strong> Rank engines by score and inspect from the top down as far as your capacity allows. Choose the threshold from your own cost of a missed case. A human reviews every flagged engine.</div>` +
      `<p>${saving > 0.005 ? `If a missed case costs about the same as an unnecessary inspection, model-guided inspection (best threshold ${f2(b1.threshold)}) lowers expected cost by about <strong>${pct(saving, 0)}</strong> versus the better no-model option.` : 'Under equal costs the model does not clearly beat the no-model options.'} ${rx ? `Once a missed case costs about <strong>${rx}×</strong> an unnecessary inspection or more, simply inspecting every engine is about as cheap.` : ''} <span class="muted">Assumptions, not company data.</span></p>`;
    $('foot-model').textContent = `Model: ${art.meta.name} · trained on ${art.meta.trained_on} (${art.meta.n_dev.toLocaleString()} rows) · exported ${art.meta.created} · ${art.meta.pygam_version ? 'pyGAM ' + art.meta.pygam_version : 'scikit-learn ' + art.meta.sklearn_version}`;
  }

  /* ---------- Predict ---------- */
  function buildForm() {
    $('fields').innerHTML = F.map((f, i) => {
      const r = art.ranges[f];
      return `<div class="field"><label for="in-${i}">${esc(f)} (${esc(U[i])})</label>` +
        `<input id="in-${i}" name="${esc(f)}" type="number" step="any" inputmode="decimal" required aria-describedby="rng-${i}">` +
        `<span class="hint" id="rng-${i}">Training range ${r.min.toFixed(r.max < 20 ? 2 : 1)} to ${r.max.toFixed(r.max < 20 ? 2 : 1)}</span></div>`;
    }).join('');
    // One-click examples: real held-out test rows, classified by the trained model itself at the default threshold.
    const realRows = art.reference.X.slice(0, art.reference.n_real_test_rows);
    const picks = C.pickExamples(model, realRows, DEFAULTS.threshold);
    const exDefs = [['hi', 'Above threshold'], ['mid', 'Near threshold'], ['lo', 'Below threshold']];
    $('examples').innerHTML = exDefs.filter((d) => picks[d[0]]).map((d) =>
      `<button type="button" class="btn ex ${d[0]}" data-ex="${d[0]}">${d[1]} <small>model says ${pct(picks[d[0]].p, 0)}</small></button>`).join('');
    $('example-note').textContent = `Examples are real rows from the held-out test set (labels not shown), sorted by the trained model at the default ${pct(DEFAULTS.threshold, 0)} threshold. Clicking one sets that threshold.`;
    $('examples').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-ex]');
      if (!b || !picks[b.dataset.ex]) return;
      const row = realRows[picks[b.dataset.ex].index];
      setThreshold(DEFAULTS.threshold);
      row.forEach((v, i) => { $('in-' + i).value = Number(v.toFixed(4)); $('in-' + i).removeAttribute('aria-invalid'); });
      $('form-errors').hidden = true;
      $('form').requestSubmit();
    });
    $('btn-clear').onclick = () => { setTimeout(() => { state.lastRow = null; $('form-errors').hidden = true; $('result').innerHTML = RESULT_PLACEHOLDER; }, 0); };
    $('form').addEventListener('submit', (e) => {
      e.preventDefault();
      const raw = {};
      F.forEach((f, i) => { raw[f] = $('in-' + i).value; });
      const v = C.validateRow(model, raw);
      F.forEach((f, i) => {
        const bad = v.errors.some((m) => m.startsWith(f + ' '));
        if (bad) $('in-' + i).setAttribute('aria-invalid', 'true'); else $('in-' + i).removeAttribute('aria-invalid');
      });
      const box = $('form-errors');
      if (!v.ok) {
        box.hidden = false;
        box.innerHTML = '<strong>Please fix the following:</strong><ul>' + v.errors.map((m) => `<li>${esc(m)}</li>`).join('') + '</ul>';
        return;
      }
      box.hidden = true;
      state.lastRow = {values: v.values, warnings: v.warnings};
      renderResult();
      $('result').scrollIntoView({block: 'start'});
    });
  }

  function renderResult() {
    const {values} = state.lastRow;
    const thr = state.threshold;
    const E = C.explainPrediction(model, values, thr);
    const p = E.p;
    const c = C.confusionAt(oof, thrIdx());
    const pctile = C.scorePercentile(oof.score_quantiles, p);
    const level = !E.reliable && E.level !== 'hi' ? 'mid' : E.level;
    const label = E.priority;
    const oddsFmt = (f) => (f > 1000 ? '>1,000×' : f < 0.001 ? '<0.001×' : f.toFixed(2) + '×');
    const items = E.items.map((it) => ({label: it.name, v: it.contribution, sub: C.withUnit(it.value, it.unit)})).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));

    const warn = E.reliable ? '' :
      `<div class="errors warn big" role="alert"><strong>Reliability warning: the model is extrapolating.</strong>` +
      `<ul>${E.outOfRange.map((i) => `<li>${esc(i.name)} = ${esc(C.withUnit(i.value, i.unit))} is outside the training range (${esc(C.withUnit(i.trainMin, i.unit))} to ${esc(C.withUnit(i.trainMax, i.unit))}).</li>`).join('')}</ul>` +
      `<p>The score below was computed as usual but may not be dependable. Verify the reading and request human review. A low score here is not a reason for routine monitoring.</p></div>`;

    const driverCard = (it) =>
      `<div class="driver ${it.direction}"><div class="dh"><strong>${esc(it.name)}</strong><span class="val">${esc(C.withUnit(it.value, it.unit))}</span>` +
      `<span class="tag ${it.direction === 'up' ? 'y' : 'n'}">${it.direction === 'up' ? '▲ raises risk' : '▼ lowers risk'} · odds ${esc(oddsFmt(it.oddsFactor))}</span></div>` +
      `<div class="dm">Development median ${esc(C.withUnit(it.median, it.unit))} · typical ${esc(C.withUnit(it.p25, it.unit))} to ${esc(C.withUnit(it.p75, it.unit))}</div>` +
      `<p>${esc(it.sentence)}</p></div>`;
    const shown = E.up.concat(E.down.slice(0, 2));
    const driverTable = shown.length
      ? `<div class="drivers">${shown.map(driverCard).join('')}</div>`
      : '<p class="muted">No sensor moved the score meaningfully; all six effects are small.</p>';

    const stepIcon = {caution: 'Verify first', area: 'Possible area to investigate', action: 'Recommended action'};
    const steps = `<ol class="steps">${E.steps.map((s) => `<li class="${s.kind}"><span class="kind">${stepIcon[s.kind]}</span>${esc(s.text)}</li>`).join('')}</ol>`;

    // At a glance: readable in about ten seconds. Everything else sits below, one click away.
    const glance =
      `<section class="panel glance ${level}" aria-labelledby="rg"><h2 id="rg" class="sr">At a glance</h2>` + warn +
      `<div class="g-top"><span class="num">${pct(p, 1)}</span><div><span class="pill ${level}">${esc(label)}</span>` +
      `<div class="g-sub">risk of the dataset's <em>Needs Maintenance</em> label · selected threshold ${pct(thr, 0)}</div></div></div>` +
      `<dl class="g-list"><div><dt>Why</dt><dd>${esc(E.glance.reason)}</dd></div><div><dt>Do this next</dt><dd>${esc(E.glance.next)}</dd></div></dl>` +
      `<p class="hint">Model estimate, not a fault diagnosis. It shows which readings to check first, not where the problem is. A person decides what to inspect and repair.</p></section>`;

    $('result').innerHTML = glance +
      `<details class="more" id="more"><summary>Show the full explanation, sensor chart, next steps and limitations</summary>` +
      // A. Assessment
      `<section class="panel" aria-labelledby="ra"><h2 id="ra">A. Maintenance assessment</h2>` +
      `<p class="muted">Predicted probability of the dataset's <em>Needs Maintenance</em> label · inspection threshold <strong>${pct(thr, 0)}</strong></p>` +
      `<div class="bar" role="img" aria-label="Risk ${pct(p, 1)}, threshold ${pct(thr, 0)}"><div class="mark" style="left:${thr * 100}%"></div><div class="dot" style="left:${Math.min(Math.max(p, 0.01), 0.99) * 100}%"></div></div>` +
      `<div class="bar-labels"><span>0%</span><span>Threshold ${pct(thr, 0)}</span><span>100%</span></div>` +
      `<div class="kv"><div><span>Recommendation</span><strong>${esc(E.priority)}</strong></div>` +
      `<div><span>Compared with development engines</span><strong>higher than ${pctile.toFixed(0)}%</strong></div>` +
      `<div><span>At this threshold (development data)</span><strong>${pct(c.precision)} of flagged engines carried the label (${pct(oof.prevalence)} by chance); ${pct(c.recall)} of all cases found</strong></div></div>` +
      `<div class="next"><strong>Recommended action.</strong> ${esc(E.actions[0])} ${E.level === 'hi' ? 'A person decides what to inspect and repair.' : ''}</div></section>` +
      // B. Why flagged
      `<section class="panel" aria-labelledby="rb"><h2 id="rb">B. ${E.level === 'hi' ? 'Why this engine was flagged' : 'Why this engine received this score'}</h2>` +
      `<p class="tag-line"><span class="tag n">Model explanation</span></p>` +
      E.narrative.map((l) => `<p>${esc(l)}</p>`).join('') + driverTable +
      `<p class="hint">Odds multipliers are measured against each sensor's average effect in the development data, so 1.30× means the odds of the label are 30% higher, not that the probability rises by 30 percentage points. ${esc(E.referenceNote)}</p></section>` +
      // C + D
      `<div class="two"><section class="panel" aria-labelledby="rc"><h2 id="rc">C. Sensor contribution chart</h2>` +
      `<p class="hint">Effect of each reading on this engine's log-odds, measured against each sensor's average effect in the development data. <span style="color:#c2410c">Orange bars raise the score</span>, <span style="color:#1f4fd8">blue bars lower it</span>. Log-odds = ${f2(model.intercept)} (reference level) + the six bars, exactly.</p><div id="chart-local"></div></section>` +
      `<section class="panel" aria-labelledby="rd"><h2 id="rd">D. Recommended next steps</h2>` +
      `<p class="hint">In order of priority. These are areas to check, not a diagnosis.</p>${steps}` +
      `<div class="next"><strong>Where is the problem?</strong> The model cannot say. It only points to the readings that raised the score. Follow your own maintenance checklist or the manufacturer's procedure for those sensors. A team-approved checklist can be added here.</div>` +
      `<div class="next">Human review is required before any repair decision.</div></section></div>` +
      // E. Limitations
      `<section class="panel" aria-labelledby="re"><h2 id="re">E. Model limitations</h2><ul>` +
      `<li>This is a predictive classification model, not a fault-diagnosis system. It cannot say <em>what</em> is wrong, only how engines like this one were labeled in the dataset.</li>` +
      `<li><strong>Confirmed diagnosis:</strong> not available. The dataset has no fault descriptions, and its label definition is unknown.</li>` +
      `<li>Manufacturer safety specifications are not included in the dataset. Compare readings with the specifications for your engine.</li>` +
      `<li>Human verification is required: sensor errors, unusual operating conditions, and readings outside the training range can all mislead the model.</li></ul></section></details>`;
    if (state.detailsOpen) $('more').open = true;
    $('more').addEventListener('toggle', () => { state.detailsOpen = $('more').open; });
    barChart($('chart-local'), items, {diverging: true, fmt: (v) => (v >= 0 ? '+' : '') + v.toFixed(2), aria: 'Contribution of each sensor reading to this engine\'s risk'});
  }

    /* ---------- Fleet: predict every engine first, prioritise inspections only when needed ---------- */
  const PAGE = 100, PRIO_LIMIT = 300;
  const cardHtml = (x) => `<div class="card ${x.cls || ''}"><div class="k">${esc(x.k)}</div><div class="v">${esc(x.v)}</div><div class="s">${esc(x.s)}</div></div>`;
  const engineLabel = (r) => {
    const open = state.openRows.has(r.rowNumber);
    return `<button type="button" class="rowbtn" data-row="${r.rowNumber}" aria-expanded="${open}" title="Show which sensors drive this score and what to check first">${open ? '▾' : '▸'} ${r.id ? esc(r.id) : 'CSV row ' + r.rowNumber}</button>`;
  };
  const oddsTxt = (it) => it.oddsFactor > 1000 ? '>1,000' : it.oddsFactor < 0.001 ? '<0.001' : it.oddsFactor.toFixed(2);
  const STEP_KIND = {caution: 'Verify first', area: 'Possible area to investigate', action: 'Recommended action'};

  // Engine detail: which sensors drive this engine's score and what to check first (same model explanation as the Predict page).
  function fleetDetail(r, cols) {
    if (!state.openRows.has(r.rowNumber)) return '';
    const E = C.explainPrediction(model, r.values, state.threshold);
    const rows = E.items.slice().sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).map((it) => {
      const eff = it.direction === 'up' ? `<span class="tag m">▲ raises score · odds ×${oddsTxt(it)}</span>`
        : it.direction === 'down' ? `<span class="tag b">▼ lowers score · odds ×${oddsTxt(it)}</span>` : '<span class="tag g">small effect</span>';
      const rel = it.outsideTraining ? '<span class="tag m">outside training range</span>' : it.relation === 'within' ? 'within typical range' : it.relation + ' typical range';
      return `<tr><td class="l">${esc(it.name)}</td><td>${esc(C.withUnit(it.value, it.unit))}</td><td class="l">${rel}</td><td class="l">${eff}</td></tr>`;
    }).join('');
    const steps = E.steps.map((s) => `<li class="${s.kind}"><span class="kind">${STEP_KIND[s.kind]}</span>${esc(s.text)}</li>`).join('');
    const label = r.id ? esc(r.id) : 'CSV row ' + r.rowNumber;
    return `<tr class="detail"><td colspan="${cols}"><div class="detail-box">` +
      `<p class="d-head"><strong>${label}</strong> · predicted probability ${pct(r.risk, 1)} · ${esc(C.recommendation(r, state.threshold))}</p>` +
      `<div class="d-grid"><div><h4>Sensors behind this score</h4><table class="mini"><thead><tr><th class="l">Sensor</th><th>Reading</th><th class="l">Compared with training data</th><th class="l">Effect on this score</th></tr></thead><tbody>${rows}</tbody></table>` +
      `<p class="hint">Effects are measured against each sensor's average in the development data. Odds multipliers are not percentage points.</p></div>` +
      `<div><h4>Why</h4><p>${esc(E.glance.reason)}</p><h4>What to check first</h4><p>${esc(E.glance.next)}</p><ol class="steps">${steps}</ol></div></div>` +
      `<div class="row"><button type="button" class="btn small" data-open-predict="${r.rowNumber}">Open the full analysis on the Predict page</button></div>` +
      `<p class="hint">Model estimate, not a fault diagnosis. It shows which readings to check first, not where the problem is. A person decides what to inspect and repair.</p></div></td></tr>`;
  }
  function openInPredict(rowNumber) {
    const r = state.fleet && state.fleet.rows.find((x) => x.rowNumber === rowNumber);
    if (!r) return;
    r.values.forEach((v, i) => { $('in-' + i).value = Number(v.toFixed(4)); $('in-' + i).removeAttribute('aria-invalid'); });
    $('form-errors').hidden = true;
    location.hash = '#predict';
    $('form').requestSubmit();
  }
  const probCell = (p) => `<td><strong>${pct(p, 1)}</strong><span class="pbar" aria-hidden="true"><i style="width:${Math.max(1, Math.round(p * 100))}%"></i></span></td>`;
  const driverCell = (r) => r.driver ? `${esc(r.driver.name)} <small>▲ odds ${r.driver.oddsFactor > 1000 ? '>1,000' : r.driver.oddsFactor.toFixed(2)}×</small>` : '<span class="muted">No single driver</span>';
  const rangeNote = (r) => r.outOfRange.length ? `<span class="tag m" title="${esc('Outside training range: ' + r.outOfRange.join(', '))}">Verify readings</span>` : '';

  function buildFleet() {
    $('req-cols').textContent = F.join(', ');
    $('sample-note').textContent = '25 real rows from the held-out test set (labels removed), with sample Engine IDs, for trying the workflow.';
    const sampleCSV = () => {
      const rows = art.reference.X.slice(0, 25);
      return ['Engine ID'].concat(F).join(',') + '\n' + rows.map((r, i) => ['SAMPLE-' + String(i + 1).padStart(2, '0')].concat(r.map((v) => Number(v.toFixed(4)))).join(',')).join('\n') + '\n';
    };
    $('btn-sample').onclick = () => download('smartops_sample_fleet.csv', sampleCSV());

    // Parse, validate and score every row of the CSV text, then show all predictions.
    function loadFleetText(text, fromSample) {
      const msg = $('fleet-msg');
      const res = C.scoreCSV(model, text);
      if (res.error) {
        state.fleet = null; $('fleet-out').hidden = true;
        msg.className = 'errors'; msg.hidden = false; msg.textContent = res.error;
        renderOverview();
        return;
      }
      if (!res.rows.length) {
        state.fleet = null; $('fleet-out').hidden = true;
        msg.className = 'errors'; msg.hidden = false; msg.textContent = 'No valid rows found. First problem: ' + (res.invalid[0] ? 'row ' + res.invalid[0].rowNumber + ': ' + res.invalid[0].errors.join(' ') : 'file is empty.');
        renderOverview();
        return;
      }
      state.fleet = res; state.fleetLimit = PAGE; state.openRows = new Set();
      $('filter').value = 'all';
      const n = res.rows.length;
      msg.className = 'errors ok'; msg.hidden = false;
      msg.textContent = (fromSample ? 'Sample loaded. ' : '') + `Analyzed all ${n.toLocaleString()} valid engine(s)` + (res.invalid.length ? `; ${res.invalid.length.toLocaleString()} row(s) could not be analyzed (details below the table)` : '') + (res.idColumn ? `; Engine IDs read from "${res.idColumn}"` : '') + (res.extraColumns.length ? `; ignored other columns: ${res.extraColumns.join(', ')}` : '') + '.';
      renderFleet(); renderOverview();
      if (fromSample) $('fleet-out').scrollIntoView({block: 'start'});
    }
    $('btn-sample-load').onclick = () => loadFleetText(sampleCSV(), true);
    $('csv').addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const msg = $('fleet-msg');
      msg.hidden = true;
      const reader = new FileReader();
      reader.onerror = () => { msg.className = 'errors'; msg.hidden = false; msg.textContent = 'The file could not be read.'; };
      reader.onload = () => loadFleetText(String(reader.result), false);
      reader.readAsText(file);
    });
    $('filter').addEventListener('change', () => { state.fleetLimit = PAGE; renderFleet(); });
    $('btn-more').onclick = () => { state.fleetLimit += PAGE; renderFleet(); };
    $('btn-all').onclick = () => { state.fleetLimit = Infinity; renderFleet(); };
    $('cap').addEventListener('input', (e) => {
      const raw = e.target.value.trim();
      state.capErr = false;
      if (raw === '') state.capN = null;
      else {
        const v = Number(raw);
        if (Number.isInteger(v) && v >= 1) state.capN = v; else { state.capN = null; state.capErr = true; }
      }
      renderPrioritize();
    });
    ['fleet-table', 'prio-table'].forEach((id) => $(id).addEventListener('click', (e) => {
      const p = e.target.closest('[data-open-predict]');
      if (p) { openInPredict(Number(p.dataset.openPredict)); return; }
      if (e.target.closest('tr.detail')) return;
      const tr = e.target.closest('tr[data-row]');
      if (!tr) return;
      const key = Number(tr.dataset.row);
      if (state.openRows.has(key)) state.openRows.delete(key); else state.openRows.add(key);
      renderFleet();
      const nb = $(id).querySelector('button.rowbtn[data-row="' + key + '"]');
      if (nb) nb.focus({preventScroll: true});
    }));
    $('btn-export-all').onclick = () => {
      if (!state.fleet) return;
      download('smartops_all_predictions.csv', C.exportAllCSV(model, state.fleet.rows, state.threshold));
    };
    $('btn-export-prio').onclick = () => {
      if (!state.fleet || state.capN === null) return;
      download('smartops_prioritized_inspections.csv', C.exportPriorityCSV(model, state.fleet.rows, state.threshold, state.capN));
    };
  }

  function renderFleet() {
    const fl = state.fleet;
    if (!fl) return;
    $('fleet-out').hidden = false;
    const thr = state.threshold, rows = fl.rows;
    const S = C.fleetSummary(rows, thr);
    $('fleet-cards').innerHTML = [
      {k: 'Total Engines Analyzed', v: S.total.toLocaleString(), s: fl.invalid.length ? `${fl.invalid.length.toLocaleString()} row(s) could not be analyzed.` : 'Every valid row was scored by the model.'},
      {k: C.NEEDS_INSPECTION, v: S.needs.toLocaleString(), s: `Predicted probability at or above ${f2(thr)}.`, cls: 'accent'},
      {k: C.BELOW_THRESHOLD, v: S.below.toLocaleString(), s: 'Probability below the threshold. This is not a guarantee of good condition.'},
      {k: 'Percentage Flagged for Inspection', v: pct(S.pctFlagged, 1), s: `${S.needs.toLocaleString()} of ${S.total.toLocaleString()} engines.`}
    ].concat(S.outOfRange ? [{k: 'Engines with readings outside the training range', v: S.outOfRange.toLocaleString(), s: 'The model extrapolates for these engines. Verify the readings before relying on the score.', cls: 'warn'}] : []).map(cardHtml).join('');
    const sb = $('fleet-stack');
    sb.setAttribute('aria-label', `${S.needs} of ${S.total} engines need inspection, ${S.below} are below the threshold`);
    sb.innerHTML = `<span class="s-need" style="width:${(S.pctFlagged * 100).toFixed(2)}%"></span><span class="s-below" style="width:${((1 - S.pctFlagged) * 100).toFixed(2)}%"></span>`;
    $('id-note').textContent = fl.idColumn ? `Engine IDs come from the "${fl.idColumn}" column.` : 'No Engine ID column was found, so engines are identified by their CSV row number (the header is row 1).';
    $('id-note').textContent += ' Click an engine to see which sensors drive its score and what to check first.';

    const mode = $('filter').value;
    const shown = mode === 'above' ? rows.filter((r) => r.risk >= thr) : mode === 'below' ? rows.filter((r) => r.risk < thr) : rows;
    const visible = shown.slice(0, state.fleetLimit);
    const head = '<thead><tr><th>Risk rank</th><th class="l">Engine</th><th>Predicted probability</th><th class="l">Inspection recommendation</th><th class="l">Main model driver</th><th class="l">Notes</th></tr></thead>';
    const body = visible.map((r) => {
      const needs = r.risk >= thr;
      const tag = needs ? `<span class="tag y">${C.NEEDS_INSPECTION}</span>` : `<span class="tag g">${C.BELOW_THRESHOLD}</span>`;
      return `<tr class="clickable ${needs ? 'flag' : ''}" data-row="${r.rowNumber}"><td>${r.rank}</td><td class="l">${engineLabel(r)}</td>${probCell(r.risk)}<td class="l">${tag}</td><td class="l">${driverCell(r)}</td><td class="l">${rangeNote(r)}</td></tr>` + fleetDetail(r, 6);
    }).join('');
    $('fleet-table').innerHTML = `<caption class="sr">Predicted maintenance probability for every analyzed engine, highest first</caption>${head}<tbody>${body || '<tr><td colspan="6" class="l">No engines match this filter.</td></tr>'}</tbody>`;
    $('fleet-more').textContent = `Showing ${visible.length.toLocaleString()} of ${shown.length.toLocaleString()} engine(s). The exported CSV always contains every analyzed engine.`;
    $('btn-more').hidden = shown.length <= visible.length;
    $('btn-all').hidden = shown.length <= visible.length;
    $('btn-more').textContent = `Show ${Math.min(PAGE, shown.length - visible.length).toLocaleString()} more`;
    $('btn-all').textContent = `Show all ${shown.length.toLocaleString()}`;
    const box = $('invalid-box');
    box.hidden = fl.invalid.length === 0;
    if (fl.invalid.length) {
      $('invalid-sum').textContent = `${fl.invalid.length} row(s) could not be analyzed and were skipped. The other rows were analyzed normally.`;
      $('invalid-list').innerHTML = fl.invalid.slice(0, 50).map((x) => `<li>Row ${x.rowNumber}: ${esc(x.errors.join(' '))}</li>`).join('') + (fl.invalid.length > 50 ? `<li>…and ${fl.invalid.length - 50} more.</li>` : '');
    }
    renderPrioritize();
    fitDetail();
  }

  // Keep the open detail panels as wide as the visible table area (the table itself may scroll sideways on phones).
  function fitDetail() {
    document.querySelectorAll('.table-wrap').forEach((w) => w.style.setProperty('--wrapw', Math.max(280, w.clientWidth - 2) + 'px'));
  }
  window.addEventListener('resize', fitDetail);

  // Optional business step. Uses the scores as they are; capacity never changes a prediction.
  function renderPrioritize() {
    const fl = state.fleet;
    if (!fl) return;
    const hint = $('cap-hint'), out = $('prio-out');
    const P = C.prioritize(fl.rows, state.threshold, state.capN === null ? NaN : state.capN);
    if (state.capErr) { hint.textContent = 'Enter a whole number of 1 or more, or leave this empty to skip this step.'; out.hidden = true; return; }
    if (state.capN === null) { hint.textContent = `Optional. ${P.flagged.toLocaleString()} engine(s) are currently marked Needs Inspection. Leave empty to skip.`; out.hidden = true; return; }
    out.hidden = false;
    if (!P.flagged) {
      hint.textContent = 'No engine is marked Needs Inspection at this threshold, so there is nothing to prioritize. Lower the threshold to flag more engines.';
      $('prio-cards').innerHTML = ''; $('prio-table').innerHTML = ''; $('prio-more').textContent = ''; $('btn-export-prio').disabled = true;
      return;
    }
    $('btn-export-prio').disabled = false;
    hint.textContent = P.spare ? `Your capacity covers all ${P.flagged.toLocaleString()} flagged engine(s), with ${P.spare.toLocaleString()} spare slot(s).` : `Inspect the ${P.selected.length.toLocaleString()} highest-risk flagged engine(s) first.`;
    $('prio-cards').innerHTML = [
      {k: 'Engines Flagged for Inspection', v: P.flagged.toLocaleString(), s: `Out of ${fl.rows.length.toLocaleString()} analyzed.`},
      {k: 'Available Inspection Capacity', v: P.capacity.toLocaleString(), s: 'Entered by you.'},
      {k: 'Inspect First', v: P.selected.length.toLocaleString(), s: 'Highest predicted probability among the flagged engines.', cls: 'accent'},
      {k: 'Flagged, outside current capacity', v: P.outside.toLocaleString(), s: P.outside ? 'Still Needs Inspection. Schedule them next; they are not cleared as safe.' : 'No flagged engine is left outside your capacity.'}
    ].map(cardHtml).join('');
    const head = '<thead><tr><th>Priority</th><th>Risk rank</th><th class="l">Engine</th><th>Predicted probability</th><th class="l">Main model driver</th><th class="l">Suggested first check</th><th class="l">Notes</th></tr></thead>';
    const body = P.selected.slice(0, PRIO_LIMIT).map((r, k) =>
      `<tr class="clickable flag" data-row="${r.rowNumber}"><td>${k + 1}</td><td>${r.rank}</td><td class="l">${engineLabel(r)}</td>${probCell(r.risk)}<td class="l">${driverCell(r)}</td><td class="l wrap">${esc(C.reviewFocus(r.driver))}</td><td class="l">${rangeNote(r)}</td></tr>` + fleetDetail(r, 7)).join('');
    $('prio-table').innerHTML = `<caption class="sr">Highest-risk flagged engines selected for inspection within the entered capacity</caption>${head}<tbody>${body}</tbody>`;
    fitDetail();
    $('prio-more').textContent = P.selected.length > PRIO_LIMIT ? `Showing the first ${PRIO_LIMIT} of ${P.selected.length.toLocaleString()}. The exported list contains all of them.` : 'Predictions above are unchanged. Capacity only sets the order of inspection.';
  }

  function download(name, text) {
    const url = URL.createObjectURL(new Blob([text], {type: 'text/csv;charset=utf-8'}));
    const a = document.createElement('a');
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /* ---------- Insights ---------- */
  function renderInsights() {
    const imp = F.map((f, j) => ({label: f, v: art.importance[f].mean_abs_logit, sub: U[j]})).sort((a, b) => b.v - a.v);
    barChart($('chart-importance'), imp, {fmt: (v) => v.toFixed(2), aria: 'Average effect of each sensor on the model log-odds'});
    $('curve-sensor').innerHTML = F.map((f, j) => `<option value="${j}">${esc(f)} (${esc(U[j])})</option>`).join('');
    $('curve-sensor').onchange = drawCurve;
    drawCurve();

    const cal = art.calibration;
    lineChart($('chart-cal'), {
      series: [{pts: [[0, 0], [1, 1]], color: '#9aa6bd', dash: '5 4', w: 1.5, label: 'Perfect calibration'},
               {pts: cal.map((b) => [b.mean_pred, b.observed]), color: '#1f4fd8', label: 'Model (out-of-fold, by risk band)'}],
      points: cal.map((b) => ({x: b.mean_pred, y: b.observed, r: 3 + Math.min(6, b.n / 800)})),
      xmin: 0, xmax: 1, ymin: 0, ymax: 1, xLabel: 'Predicted probability', yLabel: 'Observed share labelled Needs Maintenance',
      xFmt: (v) => pct(v), yFmt: (v) => pct(v), aria: 'Calibration plot of predicted vs observed risk'
    });

    const diffs = art.reference.X.map((row, i) => Math.abs(model.predict(row) - art.reference.proba[i]));
    const mx = Math.max(...diffs);
    const ok = mx < 1e-8;
    $('selfcheck').innerHTML = `<div class="errors ${ok ? 'ok' : ''}"><strong>${ok ? 'Pass' : 'FAIL'}:</strong> ${art.reference.X.length} reference predictions (${art.reference.n_real_test_rows} real test rows plus edge cases outside the training range) differ from the values the training library produced by at most ${mx.toExponential(1)}.</div>`;
  }

  function drawCurve() {
    const j = Number($('curve-sensor').value || 0);
    const r = art.ranges[F[j]];
    const med = F.map((f) => art.ranges[f].median);
    const N = 120, pts = [];
    for (let i = 0; i <= N; i++) {
      const x = r.min + ((r.max - r.min) * i) / N;
      const row = med.slice(); row[j] = x;
      pts.push([x, model.predict(row)]);
    }
    const ys = pts.map((p) => p[1]);
    lineChart($('chart-curve'), {
      series: [{pts: pts, label: 'Predicted risk'}],
      vlines: [{x: r.p01, label: '1st pct'}, {x: r.median, label: 'median'}, {x: r.p99, label: '99th pct'}],
      hlines: [{y: state.threshold, label: 'threshold ' + f2(state.threshold), color: '#b45309'}],
      ymin: Math.max(0, Math.min(...ys, state.threshold) - 0.05), ymax: Math.min(1, Math.max(...ys, state.threshold) + 0.05),
      xLabel: `${F[j]} (${U[j]})`, yLabel: 'Predicted risk', xFmt: (v) => v.toFixed(r.max - r.min > 50 ? 0 : 1), yFmt: (v) => pct(v),
      aria: 'How predicted risk changes with ' + F[j]
    });
    $('curve-note').textContent = 'Other five sensors held at their development medians. Dotted lines mark where most development data lies. The curve is how the model behaves, not proof of a physical cause.';
  }

  /* ---------- Business ---------- */
  function renderBusiness() {
    const r = state.ratio, unit = state.unit, ti = thrIdx(), cap = state.capPct;
    const S = C.strategies(oof, r, ti);
    const best = C.bestThreshold(oof, r);
    const Sb = C.strategies(oof, r, best.index);
    const n = oof.n, pos = Math.round(oof.prevalence * n);
    const cp = C.captureAt(art.capture, oof, cap);
    const capCost = ((cp.n - cp.tp) + r * (pos - cp.tp)) / n;
    const rndTp = pos * (cp.n / n), rndCost = ((cp.n - rndTp) + r * (pos - rndTp)) / n;
    const usd = (x) => '$' + Math.round(x * unit * 1000).toLocaleString();
    const rows = [
      {name: `Model-guided, threshold ${f2(state.threshold)}`, insp: S.model.conf.flaggedShare, rec: S.model.conf.recall, prec: S.model.conf.precision, cost: S.model.cost},
      {name: `Model-guided, cost-optimal threshold ${f2(best.threshold)}`, insp: Sb.model.conf.flaggedShare, rec: Sb.model.conf.recall, prec: Sb.model.conf.precision, cost: Sb.model.cost},
      {name: `Top ${cap}% by risk (your capacity)`, insp: cp.n / n, rec: cp.recall, prec: cp.precision, cost: capCost},
      {name: `Random ${cap}% (no model, same capacity)`, insp: cp.n / n, rec: cp.n / n, prec: oof.prevalence, cost: rndCost},
      {name: 'Inspect all engines', insp: 1, rec: 1, prec: oof.prevalence, cost: S.all.cost},
      {name: 'Inspect none', insp: 0, rec: 0, prec: null, cost: S.none.cost}
    ];
    $('strat-table').innerHTML = `<caption class="sr">Cost of each inspection strategy</caption><thead><tr><th class="l">Strategy</th><th>Engines inspected</th><th>Cases found</th><th>Precision</th><th>Cost units / engine</th><th>Hypothetical USD per 1,000 engines</th></tr></thead><tbody>` +
      rows.map((x) => `<tr><td class="l">${esc(x.name)}</td><td>${pct(x.insp, 0)}</td><td>${pct(x.rec, 0)}</td><td>${x.prec == null ? '–' : pct(x.prec, 0)}</td><td>${x.cost.toFixed(3)}</td><td>${usd(x.cost)}</td></tr>`).join('') + '</tbody>';

    const bestNoModel = Math.min(S.all.cost, S.none.cost);
    const which = S.all.cost <= S.none.cost ? 'inspecting every engine' : 'inspecting none';
    const gain = 1 - Sb.model.cost / bestNoModel;
    $('biz-summary').innerHTML =
      `<h2>What this means</h2>` +
      `<div class="cards" style="grid-template-columns:1fr 1fr;margin-top:0">` +
      `<div class="card"><div class="k">Model, your threshold</div><div class="v">${S.model.cost.toFixed(3)}</div><div class="s">cost units per engine · ${usd(S.model.cost)} per 1,000 (hypothetical)</div></div>` +
      `<div class="card"><div class="k">Best no-model option</div><div class="v">${bestNoModel.toFixed(3)}</div><div class="s">${esc(which)} · ${usd(bestNoModel)} per 1,000</div></div></div>` +
      `<div class="next">${gain > 0.005 ? `At a miss-cost ratio of <strong>${r}×</strong>, the cost-optimal threshold (<strong>${f2(best.threshold)}</strong>) lowers expected cost by about <strong>${pct(gain, 1)}</strong> versus ${esc(which)}. That is a hypothetical saving of about ${usd(bestNoModel - Sb.model.cost)} per 1,000 engines.` : `At a miss-cost ratio of <strong>${r}×</strong>, <strong>${esc(which)}</strong> is about as cheap as using the model. The model's value here is mainly in <em>ranking</em> when you can inspect only some engines.`}</div>` +
      `<p class="hint">Why the gains are modest: the model ranks the right engine higher about ${pct(M.test.auc, 0)} of the time, and ${pct(oof.prevalence, 0)} of engines carry the label anyway, so picking at random is already right most of the time.</p>`;

    const mk = (key) => oof.thresholds.map((t, i) => [t, C.costFor(C.confusionAt(oof, i), r)]);
    const curve = mk();
    const ys = curve.map((p) => p[1]).concat([S.all.cost, S.none.cost]);
    lineChart($('chart-cost'), {
      series: [{pts: curve, label: 'Model-guided'}],
      hlines: [{y: S.all.cost, label: 'inspect all', color: '#117a3d'}, {y: S.none.cost, label: 'inspect none', color: '#b42318'}],
      vlines: [{x: state.threshold, label: 'your threshold', color: '#b45309', dash: '4 3'}],
      points: [{x: best.threshold, y: best.cost, color: '#117a3d', label: 'best ' + f2(best.threshold)}],
      xmin: 0, xmax: 1, ymin: 0, ymax: Math.min(Math.max(...ys) * 1.05, Math.max(S.all.cost, S.model.cost, best.cost) * 2.2),
      xLabel: 'Decision threshold', yLabel: 'Expected cost per engine', yFmt: (v) => v.toFixed(2), xFmt: (v) => f2(v),
      aria: 'Expected cost per engine as the threshold changes'
    });
    const capt = art.capture;
    const cpts = [[0, 0]].concat(capt.pct.map((p, i) => [p / 100, capt.tp[i] / pos]));
    lineChart($('chart-capture'), {
      series: [{pts: [[0, 0], [1, 1]], color: '#9aa6bd', dash: '5 4', w: 1.5, label: 'Random picking'}, {pts: cpts, label: 'Highest risk first'}],
      points: [{x: cap / 100, y: cp.recall, color: '#b45309', label: `${cap}% → ${pct(cp.recall, 0)} of cases`}],
      xmin: 0, xmax: 1, ymin: 0, ymax: 1, xLabel: 'Share of engines inspected', yLabel: 'Share of cases found', xFmt: (v) => pct(v), yFmt: (v) => pct(v),
      aria: 'Share of maintenance cases found as more engines are inspected'
    });
    $('capture-note').innerHTML = `Inspecting the top <strong>${cap}%</strong> finds <strong>${pct(cp.recall, 0)}</strong> of the cases, and <strong>${pct(cp.precision, 0)}</strong> of those inspected carry the label (${pct(oof.prevalence, 0)} by chance, lift ${(cp.precision / oof.prevalence).toFixed(2)}×).`;
  }

  function buildBusiness() {
    $('unit').value = state.unit;
    $('ratio').value = state.ratio; $('ratio-out').textContent = state.ratio + '×';
    $('thr-biz').value = state.threshold; $('thr-biz-out').textContent = f2(state.threshold);
    $('cap-pct').value = state.capPct; $('cap-pct-out').textContent = state.capPct + '%';
    $('unit').addEventListener('input', (e) => { const v = Number(e.target.value); state.unit = Number.isFinite(v) && v >= 0 ? v : 0; renderBusiness(); });
    $('ratio').addEventListener('input', (e) => { state.ratio = Number(e.target.value); $('ratio-out').textContent = state.ratio + '×'; renderBusiness(); });
    $('thr-biz').addEventListener('input', (e) => setThreshold(Number(e.target.value), 'thr-biz'));
    $('cap-pct').addEventListener('input', (e) => { state.capPct = Number(e.target.value); $('cap-pct-out').textContent = state.capPct + '%'; renderBusiness(); });
    $('btn-opt').onclick = () => setThreshold(C.bestThreshold(oof, state.ratio).threshold);
    $('thr-fleet').addEventListener('input', (e) => setThreshold(Number(e.target.value), 'thr-fleet'));
    $('thr-fleet').value = state.threshold; $('thr-fleet-out').textContent = f2(state.threshold);
    $('thr-pred').addEventListener('input', (e) => setThreshold(Number(e.target.value), 'thr-pred'));
    $('thr-pred').value = state.threshold; $('thr-pred-out').textContent = f2(state.threshold);
  }

  /* ---------- reset demo ---------- */
  function resetDemo() {
    Object.assign(state, {threshold: DEFAULTS.threshold, ratio: DEFAULTS.ratio, unit: DEFAULTS.unit, capPct: DEFAULTS.capPct, fleet: null, capN: null, capErr: false, fleetLimit: 100, openRows: new Set(), lastRow: null, detailsOpen: false});
    // Predict
    F.forEach((f, i) => { const e = $('in-' + i); e.value = ''; e.removeAttribute('aria-invalid'); });
    $('form-errors').hidden = true;
    $('result').innerHTML = RESULT_PLACEHOLDER;
    // Fleet
    $('csv').value = '';
    $('fleet-msg').hidden = true; $('fleet-out').hidden = true;
    $('cap').value = ''; $('cap-hint').textContent = ''; $('prio-out').hidden = true;
    $('filter').value = 'all';
    // Shared settings and Business assumptions
    ['thr-fleet', 'thr-biz', 'thr-pred'].forEach((id) => { $(id).value = DEFAULTS.threshold; });
    ['thr-fleet-out', 'thr-biz-out', 'thr-pred-out'].forEach((id) => { $(id).textContent = f2(DEFAULTS.threshold); });
    $('unit').value = DEFAULTS.unit;
    $('ratio').value = DEFAULTS.ratio; $('ratio-out').textContent = DEFAULTS.ratio + '×';
    $('cap-pct').value = DEFAULTS.capPct; $('cap-pct-out').textContent = DEFAULTS.capPct + '%';
    // Insights
    $('curve-sensor').value = '0';
    renderOverview(); renderBusiness(); drawCurve();
    const t = $('toast');
    t.textContent = 'Demo reset: inputs cleared and default settings restored.';
    t.classList.add('show');
    clearTimeout(resetDemo.timer);
    resetDemo.timer = setTimeout(() => t.classList.remove('show'), 2600);
    window.scrollTo(0, 0);
  }

  /* ---------- start ---------- */
  $('btn-reset').addEventListener('click', resetDemo);
  $('r-auc').textContent = f2(M.test.auc);
  $('r-auc-pct').textContent = pct(M.test.auc, 0);
  buildForm(); buildFleet(); buildBusiness();
  renderOverview(); renderInsights(); renderBusiness();
  window.addEventListener('hashchange', () => { if (location.hash === '#insights') drawCurve(); });
})();
