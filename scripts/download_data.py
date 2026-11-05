"""Download the predictive-maintenance dataset into data/predictive-maintenance-data/.

Source: https://huggingface.co/datasets/jskswamy/predictive-maintenance-data (MIT license)
Uses only the Python standard library, so it works before/without extra installs.

Run from the repo root:
    python scripts/download_data.py
Safe to re-run: files that already exist are skipped (use --force to re-download).
"""
import argparse
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/jskswamy/predictive-maintenance-data/resolve/main/"
FILES = ["engine_data.csv", "train.csv", "val.csv", "test.csv", "README.md"]
DEST = Path(__file__).resolve().parent.parent / "data" / "predictive-maintenance-data"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DEST / name
        if target.exists() and target.stat().st_size > 0 and not args.force:
            print(f"skip   {name} (already there)")
            continue
        print(f"get    {name} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(BASE + name, target)
        except Exception as exc:  # network problems, proxy, etc.
            print(f"FAILED: {exc}")
            return 1
        print(f"ok ({target.stat().st_size:,} bytes)")
    print(f"\nDone. Files are in: {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
