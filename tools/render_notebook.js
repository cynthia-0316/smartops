/**
 * Renders an executed .ipynb as native SmartOps-styled HTML (no iframe, no nbconvert):
 * markdown cells -> text, code cells -> code blocks (highlighted client-side), outputs -> text / tables / images.
 * Nothing is executed or altered: outputs are exactly those saved in the notebook.
 * Used by build_model_lineup_with_code.js.
 */
const fs = require('fs');

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const join = (v) => (Array.isArray(v) ? v.join('') : v || '');
const stripAnsi = (s) => s.replace(/\u001b\[[0-9;]*[A-Za-z]/g, '');

/* ---- minimal markdown (enough for the notebook's own markdown cells) ---- */
function inline(t) {
  let s = esc(t).replace(/&amp;nbsp;/g, '&nbsp;');
  s = s.replace(/`([^`]+)`/g, (m, c) => '<code>' + c + '</code>');
  s = s.replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
  s = s.replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<i>$2</i>');
  s = s.replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  s = s.replace(/(^|[\s(])(https?:\/\/[^\s<)`]+)/g, (m, p, u) => p + '<a href="' + u + '" target="_blank" rel="noopener">' + u + '</a>');
  return s;
}
function markdown(src) {
  const lines = src.replace(/\r/g, '').split('\n');
  const out = []; let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (!l.trim()) { i++; continue; }
    let m = l.match(/^(#{1,4})\s+(.*)$/);
    if (m) { const lvl = Math.min(m[1].length + 2, 5); out.push(`<h${lvl} class="nb-h">${inline(m[2])}</h${lvl}>`); i++; continue; }
    if (/^\s*\|/.test(l) && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1] || '')) {
      const cells = (r) => r.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
      const head = cells(l); i += 2; const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) { rows.push(cells(lines[i])); i++; }
      out.push('<div class="tbl"><table><thead><tr>' + head.map((h) => `<th>${inline(h)}</th>`).join('') + '</tr></thead><tbody>' +
        rows.map((r) => '<tr>' + r.map((c) => `<td>${inline(c)}</td>`).join('') + '</tr>').join('') + '</tbody></table></div>');
      continue;
    }
    if (/^\s*([*-]|\d+\.)\s+/.test(l)) {
      const ordered = /^\s*\d+\./.test(l); const items = [];
      while (i < lines.length && (/^\s*([*-]|\d+\.)\s+/.test(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && items.length))) {
        if (/^\s*([*-]|\d+\.)\s+/.test(lines[i])) items.push(lines[i].replace(/^\s*([*-]|\d+\.)\s+/, ''));
        else items[items.length - 1] += ' ' + lines[i].trim();
        i++;
      }
      out.push((ordered ? '<ol' : '<ul') + ' class="plain">' + items.map((t) => `<li>${inline(t)}</li>`).join('') + (ordered ? '</ol>' : '</ul>'));
      continue;
    }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|\s*\||\s*([*-]|\d+\.)\s)/.test(lines[i])) { para.push(lines[i].trim()); i++; }
    out.push(`<p class="nb-p">${inline(para.join(' '))}</p>`);
  }
  return out.join('\n');
}

/* ---- outputs ---- */
function dataframeTable(html) {
  const m = html.match(/<table[\s\S]*?<\/table>/i);
  if (!m) return null;
  return '<div class="tbl nb-table">' + m[0].replace(/\sborder="\d+"/, '').replace(/\sstyle="[^"]*"/g, '') + '</div>';
}
function renderOutput(o) {
  if (o.output_type === 'stream') return `<pre class="nb-pre">${esc(stripAnsi(join(o.text)))}</pre>`;
  if (o.output_type === 'error') return `<pre class="nb-pre nb-err">${esc(stripAnsi((o.traceback || []).join('\n')))}</pre>`;
  const d = o.data || {};
  if (d['image/png']) return `<img class="nb-img" alt="Chart output" src="data:image/png;base64,${join(d['image/png']).replace(/\s+/g, '')}">`;
  if (d['text/html']) { const t = dataframeTable(join(d['text/html'])); if (t) return t; }
  if (d['text/plain']) return `<pre class="nb-pre">${esc(join(d['text/plain']))}</pre>`;
  return '';
}

module.exports = function renderNotebook(file) {
  const nb = JSON.parse(fs.readFileSync(file, 'utf8'));
  const codeCells = nb.cells.filter((c) => c.cell_type === 'code');
  const errors = codeCells.flatMap((c) => (c.outputs || []).filter((o) => o.output_type === 'error')).length;
  if (errors) throw new Error('executed notebook contains error outputs: ' + errors);
  const imgs = codeCells.reduce((n, c) => n + (c.outputs || []).filter((o) => (o.data || {})['image/png']).length, 0);
  let k = 0;
  const cells = nb.cells.map((c) => {
    const src = join(c.source);
    if (c.cell_type === 'markdown') return `<div class="nb-md">${markdown(src)}</div>`;
    k++;
    const outs = (c.outputs || []).map(renderOutput).filter(Boolean).join('\n');
    const n = c.execution_count == null ? ' ' : c.execution_count;
    return `<div class="nbcell file"><div class="file-head"><span class="pill fresh">FRESH</span><span class="fn">In [${n}]</span><span class="lb">notebook code cell ${k} of ${codeCells.length}</span><span class="meta">${src.split('\n').length} lines</span><button type="button" class="copy-btn" aria-label="Copy code of cell ${k}">Copy code</button></div>` +
      `<pre tabindex="0"><code class="nb-code">${esc(src)}</code></pre>` +
      (outs ? `<div class="nb-out"><span class="lbl">Output (saved by Colab)</span>${outs}</div>` : (src.trim().startsWith('%pip') ? '<div class="nb-out"><span class="lbl">Output</span><pre class="nb-pre">(installs quietly, no output)</pre></div>' : '')) + '</div>';
  }).join('\n');
  return `<div class="card" id="nb-card">
    <div class="fam-top"><h3>Executed Colab notebook: code and real outputs</h3><span class="pill fresh">EXECUTED IN COLAB</span></div>
    <p class="note">Every cell in the order it ran: ${codeCells.length} code cells, ${imgs} charts, no error outputs. Tables, charts and printed text are the outputs Colab saved; nothing was rerun or edited after the run. Sections A to M below show the same code by topic, plus the saved historical scripts.</p>
    <div class="nb-cells">${cells}</div>
  </div>`;
};
