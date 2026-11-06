"""Shared evaluation protocol for SmartOps (issue #8).

Every model in the project is trained and scored through this module so that
results are comparable.

Protocol
--------
* Data: Hugging Face splits in data/predictive-maintenance-data/
  (train 14,650 / val 1,954 / test 2,931 rows, stratified, no overlap).
* Development set = train + val (16,604 rows). All model selection and tuning
  happens here with repeated stratified 5-fold CV (3 repeats, seed 42).
* The test set is scored ONCE, at the end (issue #18). ``load_test`` refuses to
  load it unless you pass ``confirm=True``.
* Metrics: ROC AUC (main), PR AUC, F1, recall and precision for the
  "Needs Maintenance" class at the default 0.5 threshold. Threshold tuning is
  done separately in issue #15.
* Every run is appended to results/model_results.csv.

Typical use (from a notebook in notebooks/ or a script in the repo root)::

    from src.common import load_dev, evaluate
    X, y = load_dev()
    from sklearn.ensemble import HistGradientBoostingClassifier
    evaluate(HistGradientBoostingClassifier(random_state=42), X, y,
             name="HistGradientBoosting", params={"defaults": True})
"""
from __future__ import annotations

import getpass
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "predictive-maintenance-data"
RESULTS_FILE = ROOT / "results" / "model_results.csv"

TARGET = "Engine Condition"
RAW_FEATURES = [
    "Engine RPM",
    "Lub Oil Pressure",
    "Fuel Pressure",
    "Coolant Pressure",
    "Lub Oil Temp",
    "Coolant Temp",
]
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3

LOG_COLUMNS = [
    "date", "who", "model", "features", "params", "n_rows",
    "cv_auc_mean", "cv_auc_std", "cv_pr_auc_mean", "cv_pr_auc_std",
    "cv_f1_mean", "cv_f1_std", "cv_recall_mean", "cv_recall_std",
    "cv_precision_mean", "cv_precision_std", "fit_time_mean_s", "notes",
]
SCORING = {
    "auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
}


def _read(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_data.py"
        )
    return pd.read_csv(path)


def _select(df: pd.DataFrame, features: str) -> tuple[pd.DataFrame, pd.Series]:
    """features='raw' -> 6 sensors; features='all' -> sensors + engineered columns."""
    if features == "raw":
        cols = RAW_FEATURES
    elif features == "all":
        cols = [c for c in df.columns if c != TARGET]
    else:
        raise ValueError("features must be 'raw' or 'all'")
    return df[cols].copy(), df[TARGET].copy()


def load_dev(features: str = "raw") -> tuple[pd.DataFrame, pd.Series]:
    """Development data (train + val) for CV, tuning and model selection."""
    dev = pd.concat([_read("train.csv"), _read("val.csv")], ignore_index=True)
    return _select(dev, features)


def load_train_val(features: str = "raw"):
    """Official train and val separately, e.g. for early stopping.

    Returns (X_train, y_train, X_val, y_val).
    """
    X_tr, y_tr = _select(_read("train.csv"), features)
    X_va, y_va = _select(_read("val.csv"), features)
    return X_tr, y_tr, X_va, y_va


def load_test(features: str = "raw", confirm: bool = False):
    """Held-out test set. Only for the final evaluation (issue #18)."""
    if not confirm:
        raise RuntimeError(
            "The test set is scored once, at the end (issue #18). "
            "Pass confirm=True only for the final comparison."
        )
    return _select(_read("test.csv"), features)


def get_cv() -> RepeatedStratifiedKFold:
    """The one CV scheme everybody uses: 5-fold x 3 repeats, seed 42."""
    return RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )


def _who() -> str:
    try:
        name = subprocess.run(
            ["git", "config", "user.name"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        if name:
            return name
    except Exception:
        pass
    return getpass.getuser()


def log_result(row: dict, path: Path = RESULTS_FILE) -> None:
    """Append one row to the shared results CSV (creates it with a header if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame([{c: row.get(c, "") for c in LOG_COLUMNS}], columns=LOG_COLUMNS)
    out.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0,
               index=False)


def evaluate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
    params: dict | None = None,
    features: str = "raw",
    notes: str = "",
    log: bool = True,
    n_jobs: int = -1,
) -> dict:
    """Score ``model`` with the shared CV and (by default) log the result.

    ``model`` can be a scikit-learn estimator or Pipeline (also XGBoost,
    LightGBM, CatBoost scikit-learn wrappers). Scaling or other preprocessing
    must be inside a Pipeline so it is fit on each training fold only.
    Returns a dict with the mean and std of each metric.
    """
    res = cross_validate(
        clone(model), X, y, cv=get_cv(), scoring=SCORING, n_jobs=n_jobs,
        return_train_score=False,
    )
    row = {
        "date": date.today().isoformat(),
        "who": _who(),
        "model": name,
        "features": features,
        "params": str(params) if params else "",
        "n_rows": len(X),
        "fit_time_mean_s": round(float(np.mean(res["fit_time"])), 3),
        "notes": notes,
    }
    for key in SCORING:
        scores = res[f"test_{key}"]
        row[f"cv_{key}_mean"] = round(float(np.mean(scores)), 4)
        row[f"cv_{key}_std"] = round(float(np.std(scores)), 4)
    if log:
        log_result(row)
    return row
