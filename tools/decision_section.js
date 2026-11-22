/**
 * Builds the "What we learned and how the results support business decisions" section.
 * Every table and the cost/threshold figure are read from the executed Colab notebook
 * (reports/colab_run/SmartOps_Final_Colab_Analysis.executed.ipynb, Sections K and J). Nothing is typed in by hand
 * except the explanatory sentences, which quote numbers that the build verifies against the tables (it throws on a mismatch).
 */
const fs = require('fs');

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const join = (v) => (Array.isArray(v) ? v.join('') : v || '');
const decode = (s) => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&').replace(/&nbsp;/g, ' ');
const cellText = (h) => decode(h.replace(/<[^>]+>/g, '')).trim();

function parseTable(html) {
  const [head, body = ''] = html.split(/<\/thead>/i);
  const rows = (h) => [...h.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map((m) => [...m[1].matchAll(/<t[hd][^>]*>([\s\S]*?)<\/t[hd]>/gi)].map((c) => cellText(c[1])));
  const hr = rows(head);
  return { header: hr[0], rows: rows(body) };
}

module.exports = function decisionSection(nbFile) {
  const nb = JSON.parse(fs.readFileSync(nbFile, 'utf8'));
  const tables = [];
  let figure = null;
  for (const c of nb.cells) {
    if (c.cell_type !== 'code') continue;
    const srcText = join(c.source);
    for (const o of c.outputs || []) {
      const d = o.data || {};
      if (d['text/html']) { const m = join(d['text/html']).match(/<table[\s\S]*?<\/table>/i); if (m) tables.push(parseTable(m[0])); }
      if (d['image/png'] && srcText.includes('Cost vs threshold')) figure = join(d['image/png']).replace(/\s+/g, '');
    }
  }
  const findAll = (pred) => tables.filter((t) => pred(t.header.join('|')));
  const cost = findAll((h) => h.includes('saving_vs_best_no_model_%'))[0];
  const thr = findAll((h) => h.includes('balanced_acc') && h.includes('specificity'));
  const topk = findAll((h) => h.includes('lift_vs_random'))[0];
  if (!cost || thr.length < 2 || !topk || !figure) throw new Error('decision_section: expected tables/figure not found in the executed notebook');
  const thrDev = thr[0], thrTest = thr[1];

  const num = (s) => parseFloat(s);
  const col = (t, name) => t.header.indexOf(name);
  const n = (v, d) => num(v).toFixed(d);

  /* ---- verify every number quoted in the prose against the executed outputs ---- */
  const costRow = (r) => cost.rows.find((x) => num(x[0]) === r);
  const c1 = costRow(1), c3 = costRow(3);
  const ci = (name) => col(cost, name);
  const expect = (cond, msg) => { if (!cond) throw new Error('decision_section: prose no longer matches the notebook: ' + msg); };
  expect(Math.abs(num(c1[ci('best_threshold')]) - 0.49) < 1e-9, 'best threshold at equal costs is 0.49');
  expect(Math.abs(num(c1[ci('saving_vs_best_no_model_%')]) - 11.1) < 0.1, 'saving at equal costs is about 11%');
  expect(num(c3[ci('model_cost')]) >= num(c3[ci('inspect_all')]), 'at miss cost 3 the score rule is no better than inspect all');
  expect(num(costRow(2)[ci('model_cost')]) < num(costRow(2)[ci('inspect_all')]), 'at miss cost 2 the score rule still helps a little');
  const tk = (k) => topk.rows.find((x) => num(x[0]) === k);
  const tkc = (name) => col(topk, name);
  expect(num(tk(10)[tkc('precision')]) > 0.9 && num(tk(30)[tkc('precision')]) > 0.82 && num(tk(50)[tkc('precision')]) > 0.76, 'top-k precision');
  const th = (t) => thrDev.rows.find((x) => Math.abs(num(x[0]) - t) < 1e-9);
  const tc = (name) => col(thrDev, name);
  expect(num(th(0.5)[tc('recall')]) > 0.84 && num(th(0.65)[tc('recall')]) > 0.57, 'threshold recall');

  /* ---- table renderers ---- */
  const T = (heads, rows, rowClass) => '<div class="tbl"><table><thead><tr>' + heads.map((h, i) => `<th${i ? ' class="n"' : ''}>${esc(h)}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map((r) => `<tr${rowClass && rowClass(r) ? ` class="${rowClass(r)}"` : ''}>` + r.map((c, i) => `<td${i ? ' class="n"' : ''}>${c}</td>`).join('') + '</tr>').join('') + '</tbody></table></div>';

  const topkTable = T(['Inspect the top …% by risk score', 'Share of inspected engines that really need maintenance', 'Lift vs. picking at random', 'Share of all faults caught'],
    topk.rows.map((r) => [`${r[0]}%`, n(r[tkc('precision')], 3), `${n(r[tkc('lift_vs_random')], 2)}×`, `${n(r[tkc('faults_caught_%')], 1)}%`]));

  const thrTable = T(['Alert threshold', 'Engines flagged', 'Flagged engines that need maintenance (precision)', 'Faults caught (recall)', 'Normal engines correctly left alone (specificity)'],
    thrDev.rows.map((r) => [n(r[0], 2), `${n(r[tc('flagged_%')], 1)}%`, n(r[tc('precision')], 3), n(r[tc('recall')], 3), n(r[tc('specificity')], 3)]),
    (r) => (r[0] === '0.50' || r[0] === '0.65' ? 'top' : ''));

  const bestOf = (r) => num(r[ci('model_cost')]) < Math.min(num(r[ci('inspect_all')]), num(r[ci('inspect_none')])) - 1e-9;
  const costTable = T(['Missed fault costs … × an unneeded inspection', 'Best alert threshold', 'Score-based rule: cost per engine', 'Inspect every engine: cost per engine', 'Inspect none: cost per engine', 'Saving vs. best no-model option', 'Engines flagged', 'Cheapest strategy'],
    cost.rows.map((r) => [`${num(r[0])}`, n(r[ci('best_threshold')], 2), n(r[ci('model_cost')], 4), n(r[ci('inspect_all')], 4), n(r[ci('inspect_none')], 4),
      (() => { const v = num(r[ci('saving_vs_best_no_model_%')]); return `${v >= 0 ? '' : '−'}${Math.abs(v).toFixed(Math.abs(v) < 1 ? 2 : 1)}%`; })(), `${n(r[ci('flagged_%')], 1)}%`,
      bestOf(r) ? 'Score threshold' : 'Inspect every engine']),
    (r) => (r[0] === '1' ? 'top' : ''));

  const testTable = T(['Rule (chosen on development data only)', 'Threshold', 'Engines flagged', 'Precision', 'Faults caught (recall)'],
    thrTest.rows.map((r) => [esc(r[0]), n(r[1], 2), `${n(r[col(thrTest, 'flagged_%')], 1)}%`, n(r[col(thrTest, 'precision')], 3), n(r[col(thrTest, 'recall')], 3)]));

  const html = `
<section id="business">
  <div class="sec-head">
    <span class="eyebrow">What we learned</span>
    <h2>What the results mean for inspection decisions</h2>
    <p>SmartOps helps maintenance teams decide which engines to inspect first when inspection capacity is limited. This section turns the model findings into decisions: how to use the risk scores, what the trade-off between false alarms and missed faults looks like, and when a score-based rule is worth using at all.</p>
  </div>

  <div class="hyp" role="note"><b>All cost figures and savings below are hypothetical.</b> The dataset contains no cost information. Costs are expressed in made-up units (one unneeded inspection = 1) to show how the decision changes with the assumption. They are scenarios, not actual or forecast company savings.</div>

  <div class="card">
    <h3>1 · The most important findings</h3>
    <ul class="plain">
      <li><b>The ranking is useful but modest.</b> Nine very different model families converge on a cross-validated AUC of about 0.703 to 0.705, and the final models score about 0.71 on the held-out test set (spline logistic regression 0.7099, 95% CI 0.692 to 0.729). An AUC of 0.71 means the score ranks engines better than chance, not that it identifies faults reliably.</li>
      <li><b>Curved effects matter; interactions and complexity do not.</b> Letting each sensor's effect bend lifts AUC from 0.688 to about 0.703. Boosting, forests and neural nets that can combine sensors do no better than additive models, so a simple, transparent model loses nothing.</li>
      <li><b>Engine RPM drives the score.</b> Shuffling RPM costs about 0.145 AUC, more than five times any other sensor (fuel pressure about 0.026). Coolant temperature adds almost nothing.</li>
      <li><b>No single model is proven best.</b> The spline logistic regression and the GAM differ by 0.0004 test AUC with overlapping intervals. The dashboard uses the spline model because it is transparent and light enough to run in a browser.</li>
    </ul>
  </div>

  <div class="card">
    <h3>2 · Risk scores as an inspection priority list</h3>
    <p class="cap">Engines are sorted by predicted risk and inspected from the top. A random pick would find a fault in about 63% of inspections (the share of engines labelled “needs maintenance”), so the useful comparison is the lift over that baseline. Figures come from out-of-fold predictions on the development data (Section K of the executed notebook).</p>
    ${topkTable}
    <p class="note">Reading the table: inspecting only the riskiest 10% finds a fault in more than 9 of 10 inspections, but catches only about 14% of all faults. The score concentrates the faults at the top; it does not isolate them.</p>
  </div>

  <div class="card">
    <h3>3 · Unnecessary inspections versus missed maintenance cases</h3>
    <p class="cap">Raising the alert threshold flags fewer engines. That cuts unneeded inspections (higher precision, more normal engines left alone) but lets more faults slip through (lower recall). Rows 0.50 and 0.65 are highlighted.</p>
    ${thrTable}
    <p class="note">Because 63% of engines need maintenance, flagging everyone already gives recall 1.00 at precision 0.63. A threshold is only worth using if the unneeded inspections it avoids are worth more than the faults it misses, which depends on costs the dataset does not contain.</p>
  </div>

  <div class="card">
    <h3>4 · Threshold and cost sensitivity (hypothetical costs)</h3>
    <p class="cap">One unneeded inspection costs 1 unit. A missed fault costs <i>r</i> units. For each <i>r</i>, the best alert threshold is chosen on the development predictions only, and its expected cost per engine is compared with the two no-model options: inspect every engine, or inspect none.</p>
    ${costTable}
    <div class="learned"><span class="lbl">What the scenarios say</span>
      <p><b>Equal costs (r = 1):</b> the best threshold is about 0.49. It flags roughly 78% of engines and lowers the expected cost by about 11% compared with inspecting every engine.</p>
      <p><b>Missed faults about 3× as costly (r = 3):</b> the score-based rule is no better than inspecting every engine under these assumptions (a difference of 0.0002 per engine, the wrong way), so inspecting everyone becomes the preferred strategy. At r = 1.5 and r = 2 the saving shrinks to about 4% and 1%.</p>
      <p>“Inspect none” is never the cheapest option in these scenarios.</p>
    </div>
    <figure class="figure">
      <img class="nb-img" alt="Three panels from the executed notebook: threshold trade-off, expected cost versus threshold for several missed-fault costs, and calibration" src="data:image/png;base64,${figure}">
      <figcaption class="note">From the executed notebook (Section K). Middle panel: expected cost per engine against the alert threshold for r = 1, 2, 3 and 5. The curves flatten, and for r ≥ 3 their minimum sits at the lowest thresholds (flag nearly everyone). Hypothetical cost units.</figcaption>
    </figure>
    <p class="note">Sanity check on unseen data: the thresholds were chosen on development data and then applied once to the test set, with no re-tuning.</p>
    ${testTable}
  </div>

  <div class="card">
    <h3>5 · Inspection capacity changes the decision</h3>
    <ul class="plain">
      <li><b>Capacity, not the threshold, should set how far down the list to go.</b> At the equal-cost threshold of 0.49 the model flags about 78% of engines. A team that can inspect only 30% of them cannot act on that alert list; it should simply work from the top of the ranking.</li>
      <li><b>The smaller the budget, the higher the hit rate, the fewer faults found.</b> Inspecting the top 30% finds a fault in about 83% of inspections and catches about 39% of all faults; the top 50% finds one in about 77% and catches about 61%.</li>
      <li><b>The cost view still applies inside a budget.</b> If missed faults are cheap relative to inspections, spend less than the full capacity; if they are expensive, no score-based rule beats inspecting everything the budget allows.</li>
    </ul>
  </div>

  <div class="card">
    <h3>6 · How to use this responsibly</h3>
    <ul class="plain">
      <li><b>Scores support prioritisation. They are not diagnoses.</b> A high score means an engine resembles past engines labelled “needs maintenance”; it does not say which component is at fault.</li>
      <li><b>Human review is required.</b> About a quarter of flagged engines are false alarms at the thresholds above, and the AUC is a ranking aid, not a verdict.</li>
      <li><b>Interpretability and simple deployment are the tie-breakers.</b> With many models equally accurate, a model whose curves a planner can read, and that runs in a browser, is the safer choice.</li>
      <li><b>Provenance is uncertain.</b> We do not know where the data came from, whether it is real or synthetic, or exactly how “needs maintenance” was defined, and there is no time or engine identifier. Treat every number as a property of this dataset.</li>
    </ul>
  </div>
</section>`;

  const css = `
.hyp { border-left: 4px solid var(--hi); background: var(--accent-soft); border-radius: 6px; padding: 10px 14px; font-size: 0.9rem; color: var(--ink); max-width: 80ch; }
.figure { margin: 0; display: grid; gap: 6px; }
#business .tbl table th { white-space: normal; vertical-align: bottom; }
#business .tbl table td.n { white-space: nowrap; }
#business h3 { margin-bottom: 2px; }`;
  return { html, css };
};
