"""Build reports/smartops_results.html from report_template.html + report_data.json (run collect_report_data.py first)."""
from pathlib import Path
here = Path(__file__).resolve().parent
tpl = (here / "report_template.html").read_text()
data = (here / "report_data.json").read_text().replace("</", "<\\/")
body = tpl.replace("__DATA__", data)
head_end = body.index("</style>") + len("</style>")
html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + body[:head_end] + "\n</head>\n<body>\n" + body[head_end:] + "\n</body>\n</html>\n")
out = here.parent / "reports" / "smartops_results.html"
out.write_text(html)
print(out, f"{len(html)/1e6:.2f} MB")
