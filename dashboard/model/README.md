# Model files

`model.json` and `model_data.js` are produced by `notebooks/dashboard_model_export.ipynb` (run it in Google Colab, then download both files into this folder).

- `model_data.js` is what the web page loads (it works from `file://` and on Vercel).
- `model.json` is the same content, read by the tests in `../tests/`.

Do not edit these files by hand and do not commit a stand-in or hand-written model: the dashboard shows these numbers as the validated model.
