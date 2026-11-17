"""Partial-dependence risk curves for one representative model per family -> model_curves.json.

For each family, fit the model on the full development set, then for each sensor sweep it across its
1st-99th percentile range while every other sensor keeps its real values (on a 1,500-row sample),
and average the predicted probability of "Needs Maintenance". Also store the observed rate per
sensor bin as a reference. Used by build_models_page.py.

Run from experiments/:  ../.venv/bin/python model_curves.py
"""
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from harness import X, YV, EXP
from models import KDENaiveBayes, spline_lr, BP
exec(open(EXP / "batch5.py").read().split("for ns in")[0])          # GAM class used in the experiments

N_GRID, N_SAMPLE, N_BINS = 40, 1500, 20
QN = lambda: QuantileTransformer(output_distribution="normal", random_state=0)
MODELS = {   # family name (as on the page) -> (run name, factory)
    "Linear": ("lr_linear", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
    "Naive Bayes": ("kde_nb", lambda: KDENaiveBayes()),
    "k-nearest neighbours": ("knn_quantile_k600", lambda: make_pipeline(QuantileTransformer(random_state=0), KNeighborsClassifier(600, weights="distance"))),
    "Tree ensembles (RF / ET)": ("et_leaf30", lambda: ExtraTreesClassifier(500, min_samples_leaf=30, max_features=0.7, random_state=42, n_jobs=-1)),
    "Neural network": ("mlp_128x64_a1.0", lambda: make_pipeline(QN(), MLPClassifier((128, 64), alpha=1.0, early_stopping=True, n_iter_no_change=20, max_iter=1000, random_state=42))),
    "Gradient boosting": ("ref_catboost_tuned", lambda: CatBoostClassifier(**BP("best_params_catboost.json"), random_seed=42, verbose=0)),
    "Additive (GAM / splines)": ("pygam_ns20", lambda: GAM(20)),
    "Spline LR": ("lr_spline_k6_C0.03", lambda: spline_lr(6, C=0.03)),
}

rng = np.random.default_rng(0)
S = X.iloc[rng.choice(len(X), N_SAMPLE, replace=False)].reset_index(drop=True)
grids = {c: np.linspace(*np.quantile(X[c], [0.01, 0.99]), N_GRID) for c in X.columns}

out = {"sensors": list(X.columns), "grid": {c: np.round(g, 3).tolist() for c, g in grids.items()},
       "base_rate": float(YV.mean()), "observed": {}, "curves": {}}
for c in X.columns:
    edges = np.unique(np.quantile(X[c], np.linspace(0, 1, N_BINS + 1)))
    idx = np.clip(np.digitize(X[c], edges[1:-1]), 0, len(edges) - 2)
    out["observed"][c] = [{"x": round(float(X[c][idx == b].median()), 3), "rate": round(float(YV[idx == b].mean()), 4), "n": int((idx == b).sum())}
                          for b in range(len(edges) - 1) if (idx == b).any()]

for fam, (name, factory) in MODELS.items():
    m = factory().fit(X, YV)
    cur = {}
    for c in X.columns:
        Z = S.loc[np.repeat(S.index, N_GRID)].reset_index(drop=True)
        Z[c] = np.tile(grids[c], N_SAMPLE)
        p = m.predict_proba(Z)[:, 1].reshape(N_SAMPLE, N_GRID).mean(0)
        cur[c] = np.round(p, 4).tolist()
    out["curves"][fam] = {"run": name, "y": cur}
    print(f"{fam:28s} {name:22s} RPM curve {cur['Engine RPM'][0]:.3f} -> {cur['Engine RPM'][-1]:.3f}", flush=True)

(EXP / "model_curves.json").write_text(json.dumps(out))
print("wrote", EXP / "model_curves.json")
