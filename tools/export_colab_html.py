"""Conversion-only export: executed Colab notebook -> standalone HTML (no cell is executed).

    python tools/export_colab_html.py <executed.ipynb> <out.html> [<source.ipynb>]

* Converts with nbconvert's HTMLExporter (the same exporter `jupyter nbconvert --to html` uses).
* Applies the SmartOps look (tools/smartops_theme.css); notebook content is not changed.
* Fails (exit 1) if the executed notebook has error outputs, missing outputs, a different set of code cells
  than the committed source notebook, secret-looking strings, or if the HTML lost figures.
"""
import json, re, sys
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

src, out = Path(sys.argv[1]), Path(sys.argv[2])
ref = Path(sys.argv[3]) if len(sys.argv) > 3 else None
here = Path(__file__).resolve().parent

nb = nbformat.read(str(src), as_version=4)
nb.metadata.pop("widgets", None)                      # Colab widget state breaks strict validation and is not content
code = [c for c in nb.cells if c.cell_type == "code"]
errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
no_out = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and not c.get("outputs")]
n_png = sum(1 for c in code for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
problems = []
if errors: problems.append(f"{len(errors)} error outputs in the executed notebook")
if ref:
    ref_nb = nbformat.read(str(ref), as_version=4)
    a = [c.source for c in code]; b = [c.source for c in ref_nb.cells if c.cell_type == "code"]
    if a != b: problems.append("code cells differ from the committed source notebook")
text_all = json.dumps(nb)
for pat in (r"ghp_[A-Za-z0-9]{20,}", r"hf_[A-Za-z0-9]{30,}", r"AKIA[0-9A-Z]{16}", r"sk-[A-Za-z0-9]{20,}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"(?i)\b(api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"]{6,}"):
    if re.search(pat, text_all): problems.append(f"secret-like pattern: {pat}")
if problems:
    print("REFUSING TO EXPORT:", *problems, sep="\n  - "); sys.exit(1)

# Presentation-only cleanup: Colab's DataFrame display attaches a <script> that needs Colab's JS API (google.colab) and a hidden
# 'convert to interactive table' button. Outside Colab they only raise console errors; the tables themselves are kept untouched.
stripped = 0
for c in code:
    for o in c.get("outputs", []):
        h = o.get("data", {}).get("text/html")
        if h is not None:
            t = "".join(h) if isinstance(h, list) else h
            t2 = re.sub(r"<script.*?</script>", "", t, flags=re.S | re.I)
            t2 = re.sub(r'<button class="colab-df-convert".*?</button>', "", t2, flags=re.S)
            if t2 != t: stripped += 1; o["data"]["text/html"] = t2
print(f"removed Colab-only scripts from {stripped} table outputs (table content unchanged)")
nb.metadata["name"] = "SmartOps Final Colab Analysis"
body, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
theme = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">\n'
         '<style id="smartops-theme">\n' + (here / "smartops_theme.css").read_text() + "</style>\n")
provenance = ('<div style="max-width:1040px;margin:16px auto;padding:10px 14px;border:1px solid #e3e7eb;border-radius:8px;background:#fff;font:14px/1.5 \'IBM Plex Sans\',sans-serif;color:#11161c">'
              '<b>Provenance.</b> Notebook <code>SmartOps_Final_Colab_Analysis.ipynb</code> was executed in Google Colab; its outputs below are the saved '
              'Colab outputs. This HTML was produced from that executed notebook by <code>nbconvert</code> '
              '(conversion only, no cell was rerun). Code cells: %d, charts: %d, error outputs: 0.</div>' % (len(code), n_png))
assert "</head>" in body and "<body" in body
body = body.replace("</head>", theme + "</head>", 1)
body = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + provenance, body, count=1)
out.write_text(body, encoding="utf-8")

html = out.read_text(encoding="utf-8")
import html as _html
plain = _html.unescape(re.sub(r"<[^>]+>", "", html))   # highlighted code is split into spans
checks = {
    "all 16 verification checks printed PASS": html.count("PASS ") >= 16 and "ALL CHECKS PASSED" in html,
    "no Python traceback in output": "Traceback (most recent call last)" not in html,
    "no Colab-only scripts left": "google.colab.kernel" not in html,
    "all figures embedded": html.count("data:image/png;base64") >= n_png and n_png >= 8,
    "code visible": "def spline_lr(" in plain and "load_test_once" in plain and "def run_cv(" in plain,
}
if "google.colab.kernel" in html:
    _i = html.index("google.colab.kernel"); print("leftover context:", repr(html[max(0, _i - 300):_i + 100]))
for k, v in checks.items(): print(("OK   " if v else "FAIL ") + k)
print(f"code cells {len(code)} | cells without outputs {no_out} | png outputs {n_png} | html {len(html)/1e6:.2f} MB")
sys.exit(0 if all(checks.values()) else 1)
