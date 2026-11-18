// Convert percent-format .py source into an .ipynb (no execution, no outputs).
const fs = require('fs');
const [,, src, out] = process.argv;
const lines = fs.readFileSync(src, 'utf8').replace(/\r\n/g, '\n').split('\n');
const cells = []; let cur = null;
const flush = () => { if (!cur) return; while (cur.lines.length && cur.lines[cur.lines.length-1].trim()==='') cur.lines.pop();
  const text = cur.lines.map(l => cur.type==='markdown' ? l.replace(/^# ?/, '') : l);
  const source = text.map((l,i)=> i<text.length-1 ? l+'\n' : l);
  const c = { cell_type: cur.type, metadata: {}, source };
  if (cur.type==='code') { c.execution_count = null; c.outputs = []; }
  cells.push(c); cur = null; };
for (const l of lines) {
  if (l.startsWith('# %%')) { flush(); cur = { type: l.includes('[markdown]') ? 'markdown' : 'code', lines: [] }; }
  else if (cur) cur.lines.push(l);
}
flush();
const nb = { cells, metadata: { colab: { provenance: [], name: 'SmartOps_Final_Colab_Analysis.ipynb' },
  kernelspec: { name: 'python3', display_name: 'Python 3' }, language_info: { name: 'python' } }, nbformat: 4, nbformat_minor: 5 };
cells.forEach((c,i)=>c.id='c'+String(i).padStart(3,'0'));
fs.writeFileSync(out, JSON.stringify(nb, null, 1) + '\n');
console.log('cells', cells.length, 'code', cells.filter(c=>c.cell_type==='code').length);
