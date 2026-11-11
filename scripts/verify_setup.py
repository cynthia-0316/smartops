"""Check that a teammate's setup is complete. Run from the repo root with the venv active:

    python scripts/verify_setup.py

Prints PASS/FAIL per check and exits non-zero if anything failed.
"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "predictive-maintenance-data"
EXPECTED_ROWS = {"train.csv": 14650, "val.csv": 1954, "test.csv": 2931, "engine_data.csv": 19535}

failures = 0


def check(label: str, ok: bool, hint: str = "") -> None:
    global failures
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f"  -> {hint}"))
    if not ok:
        failures += 1


check(f"Python {sys.version.split()[0]} (need 3.10+)", sys.version_info >= (3, 10),
      "install a newer Python")
check("virtual environment is active", sys.prefix != getattr(sys, "base_prefix", sys.prefix),
      "activate .venv first (see SETUP_INSTRUCTIONS.md step 7)")

for pkg in ["pandas", "numpy", "sklearn", "matplotlib", "jupyter_core",
            "xgboost", "lightgbm", "catboost", "shap", "optuna", "seaborn"]:
    try:
        importlib.import_module(pkg)
        check(f"import {pkg}", True)
    except ImportError:
        check(f"import {pkg}", False, "run: pip install -r requirements.txt")

for name, rows in EXPECTED_ROWS.items():
    path = DATA / name
    if not path.exists():
        check(f"data file {name}", False, "run: python scripts/download_data.py")
        continue
    with open(path, encoding="utf-8") as fh:
        n = sum(1 for _ in fh) - 1  # minus header
    check(f"data file {name} has {rows:,} rows (found {n:,})", n == rows,
          "re-run: python scripts/download_data.py --force")

print()
print("ALL GOOD - setup is complete." if failures == 0 else f"{failures} check(s) failed - fix them and re-run.")
sys.exit(1 if failures else 0)
