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
    document.querySelectorAll('#form button[type=submit], #btn-example, #csv, #btn-sample').forEach((e) => { e.disabled = true; });
    return;
  }

  const model = new C.Model(art);
  const F = art.features;
  const U = art.units;
  const M = art.metrics;
  const oof = art.oof;
  const state = {threshold: 0.5, ratio: 1, unit: 100, capPct: 10, fleet: null, capN: null, lastRow: null};

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
      {k: 'Model', v: 'Spline logistic regression', s: 'One additive model, fitted on development data only.', cls: 'accent', small: true},
      {k: 'Validated performance (held-out test)', v: 'AUC ' + f2(t.auc), s: `95% CI ${f2(t.auc_ci[0])} to ${f2(t.auc_ci[1])} on ${t.n.toLocaleString()} engines. Cross-validation ${f2(M.cv.auc_mean)} ± ${f2(M.cv.auc_std)}.`},
      {k: `Engines flagged at threshold ${f2(state.threshold)}`, v: pct(c.flaggedShare), s: `Development data, out-of-fold: ${Math.round(c.flaggedShare * c.n).toLocaleString()} of ${c.n.toLocaleString()} engines.`},
      {k: 'High-priority engines in your upload', v: state.fleet ? state.fleet.rows.filter((r) => r.risk >= state.threshold).length.toLocaleString() : 'None yet', s: state.fleet ? `of ${state.fleet.rows.length.toLocaleString()} valid rows above the threshold.` : 'Upload a fleet file on Fleet Priorities.'}
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
    $('foot-model').textContent = `Model: ${art.meta.name} · trained on ${art.meta.trained_on} (${art.meta.n_dev.toLocaleString()} rows) · exported ${art.meta.created} · scikit-learn ${art.meta.sklearn_version}`;
  }

  /* ---------- Predict ---------- */
  function buildForm() {
    $('fields').innerHTML = F.map((f, i) => {
      const r = art.ranges[f];
      return `<div class="field"><label for="in-${i}">${esc(f)} (${esc(U[i])})</label>` +
        `<input id="in-${i}" name="${esc(f)}" type="number" step="any" inputmode="decimal" required aria-describedby="rng-${i}">` +
        `<span class="hint" id="rng-${i}">Training range ${r.min.toFixed(r.max < 20 ? 2 : 1)} to ${r.max.toFixed(r.max < 20 ? 2 : 1)}</span></div>`;
    }).join('');
    const ex = art.reference.X[0];
    $('example-note').textContent = 'The example is a real row from the held-out test set. Its label is not shown.';
    $('btn-example').onclick = () => { ex.forEach((v, i) => { $('in-' + i).value = Number(v.toFixed(4)); $('in-' + i).removeAttribute('aria-invalid'); }); $('form-errors').hidden = true; };
    $('btn-clear').onclick = () => { setTimeout(() => { state.lastRow = null; $('form-errors').hidden = true; $('result').innerHTML = '<div class="panel"><h2>Result</h2><p class="muted">Enter readings and press <strong>Analyze Engine</strong>.</p></div>'; }, 0); };
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
    const label = {hi: 'Above threshold: recommend inspection', mid: E.reliable ? 'Near threshold: watch closely' : 'Verify readings first', lo: 'Below threshold'}[level];
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

    $('result').innerHTML =
      // A. Assessment
      `<section class="panel" aria-labelledby="ra"><h2 id="ra">A. Maintenance assessment</h2>` + warn +
      `<div class="risk"><span class="num">${pct(p, 1)}</span><span class="pill ${level}">${esc(label)}</span></div>` +
      `<p class="muted">Predicted probability of the dataset's <em>Needs Maintenance</em> label · inspection threshold <strong>${pct(thr, 0)}</strong></p>` +
      `<div class="bar" role="img" aria-label="Risk ${pct(p, 1)}, threshold ${pct(thr, 0)}"><div class="mark" style="left:${thr * 100}%"></div><div class="dot" style="left:${Math.min(Math.max(p, 0.01), 0.99) * 100}%"></div></div>` +
      `<div class="bar-labels"><span>0%</span><span>Threshold ${pct(thr, 0)}</span><span>100%</span></div>` +
      `<div class="kv"><div><span>Inspection priority</span><strong>${esc(E.priority)}</strong></div>` +
      `<div><span>Compared with development engines</span><strong>higher than ${pctile.toFixed(0)}%</strong></div>` +
      `<div><span>At this threshold (development data)</span><strong>${pct(c.precision)} of flagged engines carried the label (${pct(oof.prevalence)} by chance); ${pct(c.recall)} of all cases found</strong></div></div>` +
      `<div class="next"><strong>Recommended action.</strong> ${esc(E.actions[0])} ${E.level === 'hi' ? 'A person decides what to inspect and repair.' : ''}</div></section>` +
      // B. Why flagged
      `<section class="panel" aria-labelledby="rb"><h2 id="rb">B. ${E.level === 'hi' ? 'Why this engine was flagged' : 'Why this engine received this score'}</h2>` +
      `<p class="tag-line"><span class="tag n">Model explanation</span></p>` +
      E.narrative.map((l) => `<p>${esc(l)}</p>`).join('') + driverTable +
      `<p class="hint">Odds multipliers are relative to the average development engine, so 1.30× means the odds of the label are 30% higher, not that the probability rises by 30 percentage points. The model's baseline is ${pct(E.baseline, 0)}.</p></section>` +
      // C + D
      `<div class="two"><section class="panel" aria-labelledby="rc"><h2 id="rc">C. Sensor contribution chart</h2>` +
      `<p class="hint">Effect of each reading on this engine's log-odds versus the average development engine. <span style="color:#c2410c">Orange bars raise the score</span>, <span style="color:#1f4fd8">blue bars lower it</span>. Log-odds = ${f2(model.intercept)} (baseline) + the six bars, exactly.</p><div id="chart-local"></div></section>` +
      `<section class="panel" aria-labelledby="rd"><h2 id="rd">D. Recommended next steps</h2>` +
      `<p class="hint">In order of priority. These are areas to check, not a diagnosis.</p>${steps}` +
      `<div class="next">Human review is required before any repair decision.</div></section></div>` +
      // E. Limitations
      `<section class="panel" aria-labelledby="re"><h2 id="re">E. Model limitations</h2><ul>` +
      `<li>This is a predictive classification model, not a fault-diagnosis system. It cannot say <em>what</em> is wrong, only how engines like this one were labeled in the dataset.</li>` +
      `<li><strong>Confirmed diagnosis:</strong> not available. The dataset has no fault descriptions, and its label definition is unknown.</li>` +
      `<li>Manufacturer safety specifications are not included in the dataset. Compare readings with the specifications for your engine.</li>` +
      `<li>Human verification is required: sensor errors, unusual operating conditions, and readings outside the training range can all mislead the model.</li></ul></section>`;
    barChart($('chart-local'), items, {diverging: true, fmt: (v) => (v >= 0 ? '+' : '') + v.toFixed(2), aria: 'Contribution of each sensor reading to this engine\'s risk'});
  }

  /* ---------- Fleet ---------- */
  function buildFleet() {
    $('req-cols').textContent = F.join(', ');
    $('sample-note').textContent = '25 real rows from the held-out test set (labels removed), for trying the workflow.';
    $('btn-sample').onclick = () => {
      const rows = art.reference.X.slice(0, 25);
      const text = F.join(',') + '\n' + rows.map((r) => r.map((v) => Number(v.toFixed(4))).join(',')).join('\n') + '\n';
      download('smartops_sample_fleet.csv', text);
    };
    $('csv').addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const msg = $('fleet-msg');
      msg.hidden = true;
      const reader = new FileReader();
      reader.onerror = () => { msg.className = 'errors'; msg.hidden = false; msg.textContent = 'The file could not be read.'; };
      reader.onload = () => {
        const res = C.scoreCSV(model, String(reader.result));
        if (res.error) {
          state.fleet = null; $('fleet-out').hidden = true; $('cap').disabled = true;
          msg.className = 'errors'; msg.hidden = false; msg.textContent = res.error;
          renderOverview();
          return;
        }
        if (!res.rows.length) {
          state.fleet = null; $('fleet-out').hidden = true;
          msg.className = 'errors'; msg.hidden = false; msg.textContent = 'No valid rows found. First problem: ' + (res.invalid[0] ? 'row ' + res.invalid[0].rowNumber + ': ' + res.invalid[0].errors.join(' ') : 'file is empty.');
          return;
        }
        state.fleet = res;
        const n = res.rows.length;
        state.capN = Math.max(1, Math.ceil(n * 0.1));
        $('cap').disabled = false; $('cap').max = n; $('cap').value = state.capN;
        $('cap-hint').textContent = `Defaults to 10% of ${n.toLocaleString()} valid rows. Change it to match your workshop's capacity.`;
        msg.className = 'errors ok'; msg.hidden = false;
        msg.textContent = `Scored ${n.toLocaleString()} valid rows` + (res.invalid.length ? `; skipped ${res.invalid.length.toLocaleString()} invalid rows` : '') + (res.extraColumns.length ? `; ignored extra columns: ${res.extraColumns.join(', ')}` : '') + '.';
        renderFleet(); renderOverview();
      };
      reader.readAsText(file);
    });
    $('cap').addEventListener('input', (e) => {
      const n = state.fleet ? state.fleet.rows.length : 1;
      const v = Math.floor(Number(e.target.value));
      state.capN = Number.isFinite(v) && v >= 1 ? Math.min(v, n) : 1;
      renderFleet();
    });
    $('filter').addEventListener('change', renderFleet);
    $('btn-export').onclick = () => {
      if (!state.fleet) return;
      download('smartops_ranked_inspection_list.csv', C.exportCSV(model, state.fleet.rows, state.threshold, state.capN));
    };
  }

  function renderFleet() {
    const fl = state.fleet;
    if (!fl) return;
    $('fleet-out').hidden = false;
    const thr = state.threshold, cap = state.capN;
    const rows = fl.rows;
    const above = rows.filter((r) => r.risk >= thr);
    const inspect = rows.filter((r) => r.risk >= thr && r.rank <= cap);
    const capPct = Math.min(100, Math.max(1, Math.round((cap / rows.length) * 100)));
    const cp = C.captureAt(art.capture, oof, capPct);
    $('fleet-cards').innerHTML = [
      {k: 'Valid rows scored', v: rows.length.toLocaleString(), s: fl.invalid.length ? `${fl.invalid.length} invalid rows skipped.` : 'No invalid rows.'},
      {k: `At or above threshold ${f2(thr)}`, v: above.length.toLocaleString(), s: pct(above.length / rows.length, 1) + ' of valid rows.'},
      {k: 'Recommended for inspection now', v: inspect.length.toLocaleString(), s: `Above threshold and within your capacity of ${cap.toLocaleString()}.`, cls: 'accent'},
      {k: `Evidence for the top ${capPct}%`, v: pct(cp.precision), s: `In development data the top ${capPct}% by risk carried the label this often (${pct(cp.randomPrecision)} by chance) and found ${pct(cp.recall)} of all cases.`}
    ].concat(rows.some((r) => r.outOfRange.length) ? [{k: 'Rows with readings outside the training range', v: rows.filter((r) => r.outOfRange.length).length.toLocaleString(), s: 'The model extrapolates for these rows. Verify the readings; do not treat a low score as routine.', cls: 'warn'}] : []).map((x) => `<div class="card ${x.cls || ''}"><div class="k">${esc(x.k)}</div><div class="v">${esc(x.v)}</div><div class="s">${esc(x.s)}</div></div>`).join('');

    const mode = $('filter').value;
    let shown = rows;
    if (mode === 'above') shown = above; else if (mode === 'inspect') shown = inspect;
    const LIMIT = 200;
    const head = `<thead><tr><th>Rank</th><th>CSV row</th>${F.map((f, i) => `<th>${esc(f)}<br><small>${esc(U[i])}</small></th>`).join('')}<th>Risk</th><th class="l">Main model driver</th><th class="l">Status</th></tr></thead>`;
    const body = shown.slice(0, LIMIT).map((r) => {
      const isAbove = r.risk >= thr, within = r.rank <= cap;
      const oor = r.outOfRange.length > 0;
      const tag = isAbove && within ? '<span class="tag y">Inspect</span>' : isAbove ? '<span class="tag m">Above threshold, over capacity</span>' : oor ? '<span class="tag m">Verify readings first</span>' : '<span class="tag n">Routine</span>';
      const warnTag = oor && isAbove ? ' <span class="tag m" title="' + esc('Outside training range: ' + r.outOfRange.join(', ')) + '">Check readings</span>' : '';
      const drv = r.driver ? `${esc(r.driver.name)} <small>▲ odds ${r.driver.oddsFactor > 1000 ? '>1,000' : r.driver.oddsFactor.toFixed(2)}×</small>` : '<span class="muted">Baseline risk</span>';
      return `<tr class="${isAbove && within ? 'flag' : ''}"><td>${r.rank}</td><td>${r.rowNumber}</td>${r.values.map((v) => `<td>${Number(v.toFixed(2))}</td>`).join('')}<td><strong>${pct(r.risk, 1)}</strong></td><td class="l">${drv}</td><td class="l">${tag}${warnTag}</td></tr>`;
    }).join('');
    $('fleet-table').innerHTML = `<caption class="sr">Engines ranked by predicted maintenance risk</caption>${head}<tbody>${body || `<tr><td colspan="${F.length + 5}" class="l">No rows match this filter.</td></tr>`}</tbody>`;
    $('fleet-more').textContent = shown.length > LIMIT ? `Showing the top ${LIMIT} of ${shown.length.toLocaleString()} rows. The exported CSV contains every valid row.` : '';
    const box = $('invalid-box');
    box.hidden = fl.invalid.length === 0;
    if (fl.invalid.length) {
      $('invalid-sum').textContent = `${fl.invalid.length} invalid row(s) skipped`;
      $('invalid-list').innerHTML = fl.invalid.slice(0, 50).map((x) => `<li>Row ${x.rowNumber}: ${esc(x.errors.join(' '))}</li>`).join('') + (fl.invalid.length > 50 ? `<li>…and ${fl.invalid.length - 50} more.</li>` : '');
    }
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
    $('selfcheck').innerHTML = `<div class="errors ${ok ? 'ok' : ''}"><strong>${ok ? 'Pass' : 'FAIL'}:</strong> ${art.reference.X.length} reference predictions (${art.reference.n_real_test_rows} real test rows plus edge cases outside the training range) differ from scikit-learn by at most ${mx.toExponential(1)}.</div>`;
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

  /* ---------- start ---------- */
  $('r-auc').textContent = f2(M.test.auc);
  $('r-auc-pct').textContent = pct(M.test.auc, 0);
  buildForm(); buildFleet(); buildBusiness();
  renderOverview(); renderInsights(); renderBusiness();
  window.addEventListener('hashchange', () => { if (location.hash === '#insights') drawCurve(); });
})();
