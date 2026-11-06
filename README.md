# SmartOps

Course project for **Class 508** — predictive maintenance analytics (AI4I and hydraulic condition monitoring datasets).

## Structure

| Folder | Contents |
|---|---|
| `notebooks/` | Jupyter notebooks (modeling, data cleaning, condition monitoring) |
| `sql/` | SQL scripts |
| `reports/` | Written reports and summaries |
| `data/` | Datasets (not tracked in git; see `.gitignore`) |

## Setup

**New to this? Tell your AI assistant:** *"Follow `SETUP_INSTRUCTIONS.md` and set everything up for me."* It walks through every step (Git, Python, GitHub login, clone, packages, dataset download, verification). Manual quick version:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python scripts/download_data.py
python scripts/verify_setup.py
```

## Evaluation protocol (all models)

Every model is scored with the shared code in `src/common.py` so results are comparable. See `notebooks/00_eval_protocol_demo.ipynb`.

- **Data:** the Hugging Face splits in `data/predictive-maintenance-data/`. Development set = `train` + `val` (`load_dev()`).
- **CV:** repeated stratified 5-fold x 3, seed 42 (`evaluate()` does this for you).
- **Metrics:** ROC AUC (main), PR AUC, F1, recall, precision for "Needs Maintenance".
- **Test set:** scored once at the very end (`load_test(confirm=True)`). Do not use it for tuning.
- **Results log:** `evaluate()` appends a row to `results/model_results.csv`. Commit it with your PR (a `merge=union` rule in `.gitattributes` keeps both sides if two people add rows).
- Put scaling and other preprocessing inside a scikit-learn `Pipeline`.
