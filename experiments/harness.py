"""Experiment harness for model improvement (local work).

Same protocol as src/common.py (dev = train+val, RepeatedStratifiedKFold 5x3, seed 42)
but keeps per-fold predictions so models can be compared fold-by-fold and ensembled.

run(name, factory) -> fits factory() on each of the 15 training folds, predicts the held-out fold,
saves experiments/preds/<name>.npy  (shape 3 x n: one out-of-fold prediction per repeat),
appends to experiments/log.csv, returns per-fold AUCs.
"""
import sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from src.common import load_dev, get_cv, N_SPLITS, N_REPEATS

EXP = ROOT / "experiments"
PRED = EXP / "preds"; PRED.mkdir(parents=True, exist_ok=True)
LOG = EXP / "log.csv"

X, y = load_dev()
YV = y.values
FOLDS = list(get_cv().split(X, y))          # 15 (train_idx, test_idx), repeat r = i // 5
N = len(YV)


def _fit_one(factory, Xd, i, tr, te, fit_kw):
    m = factory()
    t = time.time()
    if fit_kw is not None:
        m.fit(Xd.iloc[tr], YV[tr], **fit_kw(Xd.iloc[tr], YV[tr]))
    else:
        m.fit(Xd.iloc[tr], YV[tr])
    ft = time.time() - t
    p = m.predict_proba(Xd.iloc[te])[:, 1]
    return i, te, p, ft


def run(name, factory, Xd=None, n_jobs=15, fit_kw=None, note=""):
    Xd = X if Xd is None else Xd
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs)(delayed(_fit_one)(factory, Xd, i, tr, te, fit_kw)
                                  for i, (tr, te) in enumerate(FOLDS))
    preds = np.full((N_REPEATS, N), np.nan)
    aucs = np.zeros(len(FOLDS)); prs = np.zeros(len(FOLDS)); fts = []
    for i, te, p, ft in out:
        preds[i // N_SPLITS, te] = p
        aucs[i] = roc_auc_score(YV[te], p); prs[i] = average_precision_score(YV[te], p); fts.append(ft)
    np.save(PRED / f"{name}.npy", preds)
    np.save(PRED / f"{name}__aucs.npy", aucs)
    row = {"name": name, "auc": aucs.mean(), "auc_std": aucs.std(), "pr_auc": prs.mean(),
           "fit_s": np.mean(fts), "wall_s": time.time() - t0, "note": note}
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    print(f"{name:55s} AUC {row['auc']:.4f} +/- {row['auc_std']:.4f}  PR {row['pr_auc']:.4f}  fit {row['fit_s']:.2f}s", flush=True)
    return aucs


def fold_aucs_from_preds(preds):
    return np.array([roc_auc_score(YV[te], preds[i // N_SPLITS, te]) for i, (tr, te) in enumerate(FOLDS)])


def load_preds(name):
    return np.load(PRED / f"{name}.npy")


def corrected_ttest(a, b):
    """Nadeau-Bengio corrected resampled t-test on paired fold scores (a - b). Returns (mean diff, p)."""
    d = np.asarray(a) - np.asarray(b); k = len(d)
    n_te = N / N_SPLITS; n_tr = N - n_te
    var = d.var(ddof=1) * (1 / k + n_te / n_tr)
    t = d.mean() / np.sqrt(var) if var > 0 else 0.0
    return d.mean(), 2 * stats.t.sf(abs(t), k - 1)


def leaderboard(ref=None):
    log = pd.read_csv(LOG).drop_duplicates("name", keep="last").sort_values("auc", ascending=False)
    if ref:
        ra = np.load(PRED / f"{ref}__aucs.npy")
        ps = []
        for n in log.name:
            try:
                ps.append(corrected_ttest(np.load(PRED / f"{n}__aucs.npy"), ra)[1])
            except FileNotFoundError:
                ps.append(np.nan)
        log["p_vs_" + ref] = ps
    return log
