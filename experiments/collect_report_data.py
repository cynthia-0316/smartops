"""Collect every number shown in reports/smartops_results.html into one JSON file."""
import warnings; warnings.filterwarnings("ignore")
import json, base64, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.inspection import permutation_importance
from harness import X, YV, load_preds, ROOT
from src.common import load_train_val, RAW_FEATURES
exec(open("batch5.py").read().split("for ns in")[0])          # GAM class
from sklearn.base import ClassifierMixin
class GAMC(ClassifierMixin, GAM):
    def fit(self, Xa, y):
        super().fit(Xa, y); self.classes_ = np.array([0, 1]); return self
    def predict(self, Xa): return (self.predict_proba(Xa)[:, 1] >= 0.5).astype(int)

D = {}
e = pd.read_csv(ROOT / "data/predictive-maintenance-data/engine_data.csv"); ye = e["Engine Condition"].values
D["dataset"] = {"rows": len(e), "pos": int(ye.sum()), "neg": int((ye == 0).sum()), "pos_share": float(ye.mean()),
                "missing": int(e.isna().sum().sum()), "dupes": int(e.duplicated().sum()),
                "splits": {"train": 14650, "val": 1954, "test": 2931}}
units = {"Engine RPM": "rpm", "Lub Oil Pressure": "bar", "Fuel Pressure": "bar", "Coolant Pressure": "bar", "Lub Oil Temp": "°C", "Coolant Temp": "°C"}
sens = []
q1, q3 = e[RAW_FEATURES].quantile(.25), e[RAW_FEATURES].quantile(.75); iqr = q3 - q1
out = (e[RAW_FEATURES] < q1 - 1.5 * iqr) | (e[RAW_FEATURES] > q3 + 1.5 * iqr)
for c in RAW_FEATURES:
    v = e[c].values; a = roc_auc_score(ye, v)
    lo, hi = np.percentile(v, [0.5, 99.5]); bins = np.linspace(lo, hi, 31)
    h0, _ = np.histogram(v[ye == 0], bins, density=True); h1, _ = np.histogram(v[ye == 1], bins, density=True)
    sens.append({"name": c, "unit": units[c], "auc": float(max(a, 1 - a)), "dir": "+" if a >= .5 else "-",
                 "mean0": float(v[ye == 0].mean()), "mean1": float(v[ye == 1].mean()), "min": float(v.min()), "max": float(v.max()),
                 "median": float(np.median(v)), "iqr_out": int(out[c].sum()), "iqr_share": float(out[c].mean()),
                 "bins": ((bins[:-1] + bins[1:]) / 2).round(4).tolist(), "h0": h0.round(6).tolist(), "h1": h1.round(6).tolist()})
D["sensors"] = sens
D["outliers"] = {"any_rows": int(out.any(axis=1).sum()), "any_share": float(out.any(axis=1).mean()),
                 "pos_share_in_outliers": float(ye[out.any(axis=1).values].mean()), "coolant_over_100": int((e["Coolant Temp"] > 100).sum())}
D["corr"] = {"names": RAW_FEATURES, "m": e[RAW_FEATURES].corr("spearman").round(3).values.tolist()}

# GAM: shapes and permutation importance
gam = GAM(20).fit(X, YV)
shapes = []
for j, c in enumerate(RAW_FEATURES):
    XX = gam.g_.generate_X_grid(term=j, n=80)
    lo, hi = np.percentile(X[c], [1, 99]); XX[:, j] = np.linspace(lo, hi, 80)
    pdep, ci = gam.g_.partial_dependence(term=j, X=XX, width=0.95)
    shapes.append({"name": c, "x": XX[:, j].round(3).tolist(), "y": pdep.round(4).tolist(), "lo": ci[:, 0].round(4).tolist(), "hi": ci[:, 1].round(4).tolist()})
D["gam_shapes"] = shapes
Xtr, ytr, Xva, yva = load_train_val()
g2 = GAMC(20).fit(Xtr, ytr.values)
pi = permutation_importance(g2, Xva, yva.values, scoring="roc_auc", n_repeats=20, random_state=42)
D["perm_gam"] = [{"name": c, "mean": float(m), "std": float(s)} for c, m, s in zip(RAW_FEATURES, pi.importances_mean, pi.importances_std)]
D["perm_catboost"] = [{"name": n, "mean": m} for n, m in [("Engine RPM", .1459), ("Fuel Pressure", .0270), ("Lub Oil Temp", .0156), ("Lub Oil Pressure", .0132), ("Coolant Pressure", .0041), ("Coolant Temp", .0005)]]

# Decision analysis on out-of-fold GAM predictions (repeat 1)
p = load_preds("pygam_ns20")[0]; prev = YV.mean()
th = []
for t in np.round(np.arange(0.40, 0.86, 0.05), 2):
    pr = p >= t; tp = (pr & (YV == 1)).sum(); fp = (pr & (YV == 0)).sum(); fn = (~pr & (YV == 1)).sum(); tn = (~pr & (YV == 0)).sum()
    prec = tp / max(tp + fp, 1); rec = tp / (tp + fn); spec = tn / (tn + fp)
    th.append({"t": float(t), "flagged": float(pr.mean()), "precision": float(prec), "recall": float(rec), "specificity": float(spec), "bal_acc": float((rec + spec) / 2)})
D["thresholds"] = th
order = np.argsort(-p); ys = YV[order]
D["topk"] = [{"k": k, "precision": float(ys[:int(len(ys) * k / 100)].mean()), "caught": float(ys[:int(len(ys) * k / 100)].sum() / ys.sum())} for k in (5, 10, 20, 30, 40, 50, 70, 100)]
grid = np.round(np.arange(0.05, 0.96, 0.01), 2); cost = []
for r in (1, 1.5, 2, 3, 5):
    cs = [((p >= t) & (YV == 0)).sum() / len(YV) + r * ((p < t) & (YV == 1)).sum() / len(YV) for t in grid]; j = int(np.argmin(cs))
    allc, nonec = 1 - prev, r * prev
    cost.append({"r": r, "t": float(grid[j]), "model": float(cs[j]), "inspect_all": float(allc), "inspect_none": float(nonec),
                 "saving": float((min(allc, nonec) - cs[j]) / min(allc, nonec)), "flagged": float((p >= grid[j]).mean())})
D["cost"] = cost
qs = np.quantile(p, np.linspace(0, 1, 11)); idx = np.clip(np.digitize(p, qs[1:-1]), 0, 9)
D["calibration"] = [{"pred": float(p[idx == b].mean()), "obs": float(YV[idx == b].mean())} for b in range(10)]
D["brier"] = {"gam_oof": float(brier_score_loss(YV, p)), "prevalence_only": float(prev * (1 - prev))}

# Leaderboard: team log + experiment log
fam = [("Dummy", "Baseline"), ("dummy", "Baseline"), ("gam", "Additive (GAM / splines)"), ("spline", "Additive (GAM / splines)"), ("spl_", "Additive (GAM / splines)"),
       ("lr_linear", "Linear"), ("LogisticRegression", "Linear"), ("ebm", "Additive (GAM / splines)"), ("nb", "Naive Bayes"), ("gnb", "Naive Bayes"),
       ("catboost", "Gradient boosting"), ("CatBoost", "Gradient boosting"), ("lgbm", "Gradient boosting"), ("LightGBM", "Gradient boosting"), ("XGBoost", "Gradient boosting"),
       ("GradientBoosting", "Gradient boosting"), ("HistG", "Gradient boosting"), ("ref_", "Gradient boosting"), ("mlp", "Neural network"), ("torch", "Neural network"),
       ("knn", "k-nearest neighbours"), ("rf_", "Tree ensembles (RF / ET)"), ("et_", "Tree ensembles (RF / ET)"), ("stack", "Ensemble / stack"), ("noisefilter", "Additive (GAM / splines)")]
def family(n):
    if n.startswith("NESTED_stack") or n.startswith("stack"): return "Ensemble / stack"
    for k, f in fam:
        if k in n: return f
    return "Other"
team = pd.read_csv(ROOT / "results/model_results.csv")
rows = [{"name": r.model + ("" if r.features == "raw" else f" [{r.features}]"), "auc": r.cv_auc_mean, "std": r.cv_auc_std, "pr": r.cv_pr_auc_mean,
         "who": "Rishabh" if "rbhand" in str(r.who) else "Uzair", "source": "Team results log", "family": family(r.model),
         "honest": ("NESTED" in str(r.notes)) or ("defaults" in str(r.params)) or ("Dummy" in r.model)} for r in team.itertuples()]
log = pd.read_csv("log.csv").drop_duplicates("name", keep="last")
for r in log.itertuples():
    rows.append({"name": r.name, "auc": r.auc, "std": r.auc_std, "pr": r.pr_auc, "who": "Uzair", "source": "Improvement experiments",
                 "family": family(r.name), "honest": r.name.startswith("NESTED") or r.name.startswith("pygam") or r.name in ("lr_linear", "gnb", "ebm_default", "ebm_no_interactions")})
D["leaderboard"] = rows

# Test set: all models scored
tp1 = pd.read_csv(ROOT / "results/test_predictions.csv"); tp2 = pd.read_csv("test_predictions_improvement.csv")
yt = tp1.y_true.values; assert (tp2.y_true.values == yt).all()
preds = {"CatBoost tuned": tp1["CatBoost tuned"].values, "LightGBM tuned": tp1["LightGBM tuned"].values, "GradientBoosting tuned": tp1["GradientBoosting (sklearn) tuned"].values,
         "Spline logistic regression": tp2["Spline LR (additive)"].values, "Stack CatBoost + spline": tp2["Stack CatBoost+Spline LR"].values, "GAM (20 splines)": tp2["GAM (pygam, 20 splines)"].values}
rng = np.random.default_rng(42); B = [rng.integers(0, len(yt), len(yt)) for _ in range(2000)]
test = []
for n, pp in preds.items():
    bs = [roc_auc_score(yt[i], pp[i]) for i in B]; fpr, tpr, _ = roc_curve(yt, pp)
    k = np.linspace(0, len(fpr) - 1, 120).astype(int)
    test.append({"name": n, "auc": float(roc_auc_score(yt, pp)), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                 "brier": float(brier_score_loss(yt, pp)), "by": "Rishabh (#18)" if n in ("CatBoost tuned", "LightGBM tuned", "GradientBoosting tuned") else "Uzair (improvement)",
                 "fpr": fpr[k].round(4).tolist(), "tpr": tpr[k].round(4).tolist()})
D["test"] = test
D["facts"] = {"ceiling": 0.7073, "ceiling_sd": 0.0040, "n_experiments": int(len(log)), "n_team_runs": int(len(team)),
              "interaction_ebm": [0.0003, 0.175], "interaction_cb": [0.0008, 0.460], "max_corr_within_class": 0.09}

# Figures (embedded)
figs = []
for f in sorted((ROOT / "reports/figures").iterdir()):
    mime = "image/svg+xml" if f.suffix == ".svg" else "image/png"
    figs.append({"file": f.name, "src": f"data:{mime};base64," + base64.b64encode(f.read_bytes()).decode()})
D["figures"] = figs
(ROOT / "experiments" / "report_data.json").write_text(json.dumps(D))
print("ok", len(rows), "leaderboard rows,", len(figs), "figures")
