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
