"""Builds reports/smartops_models.html (model lineup explainer) from report_data.json."""
import json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "report_data.json").read_text())

def desc(n):
    m = re.fullmatch(r"lr_spline_k(\d+)_C([\d.]+)(.*)", n)
    if m:
        extra = {"": "", "_winsor": ", outliers capped", "_quantile_uniform": ", rank-transformed first"}[m.group(3)]
        return f"Spline LR, {m.group(1)} knots, C = {m.group(2)}{extra}"
    m = re.fullmatch(r"spl_rpm(\d+)_others(\d+)", n)
    if m: return f"Spline LR, RPM {m.group(1)} knots, other sensors {m.group(2)}"
    m = re.fullmatch(r"pygam_ns(\d+)", n)
    if m: return f"GAM, {m.group(1)} splines per sensor, automatic smoothing"
    m = re.fullmatch(r"knn_quantile_k(\d+)", n)
    if m: return f"k-nearest neighbours, k = {m.group(1)}"
    m = re.fullmatch(r"mlp_([\dx]+)_a([\d.]+)", n)
    if m: return f"MLP, layers {m.group(1).replace('x', ' → ')}, alpha {m.group(2)}"
    m = re.fullmatch(r"catboost_depth(\d)_lr0.03_it1000", n)
    if m: return f"CatBoost, depth {m.group(1)}, 1,000 trees"
    fixed = {
        "spl_rpm10_fuel8_others5": "Spline LR, RPM 10 / fuel 8 / others 5 knots",
        "spl_k6_degree1": "Spline LR, 6 knots, straight pieces (degree 1)",
        "spl_k6_degree2": "Spline LR, 6 knots, quadratic pieces (degree 2)",
        "spl_k6_plus_rpm_target_enc": "Spline LR + RPM as a target-encoded category",
        "spl_k6_plus_tensor_Engine_FuelP": "Spline LR + RPM × fuel pressure surface",
        "spl_k6_plus_tensor_Engine_LubOi": "Spline LR + RPM × lube oil temp surface",
        "pygam_logistic_gridsearch_lam": "GAM, 12 splines per sensor, automatic smoothing",
        "pygam_ns12_quadratic": "GAM, 12 quadratic splines",
        "pygam_ns12_rpm_monotone_dec": "GAM, RPM forced to only lower risk",
        "pygam_ns12_te_rpm_oiltemp": "GAM + RPM × lube oil temp surface",
        "ebm_default": "EBM with automatic pair interactions",
        "ebm_no_interactions": "EBM without interactions",
        "NESTED_spline_lr_grid": "Spline LR, knots and C tuned inside each fold (nested)",
        "NESTED_catboost_search": "CatBoost, 15-setting search inside each fold (nested)",
        "NESTED_stack_catboost_spline": "Stack of CatBoost + spline LR, both tuned inside each fold (nested)",
        "noisefilter_q0.15_spline_lr": "Spline LR after dropping 15% most surprising rows",
        "noisefilter_q0.3_spline_lr": "Spline LR after dropping 30% most surprising rows",
        "gnb": "Gaussian naive Bayes", "gnb_quantile_normal": "Gaussian naive Bayes, rank-transformed sensors",
        "kde_nb": "KDE naive Bayes", "kde_nb_bw0.3": "KDE naive Bayes, bandwidth 0.3",
        "rf_leaf50": "Random forest, 500 trees, min 50 per leaf", "et_leaf30": "Extra trees, 500 trees, min 30 per leaf",
        "mlp_64_32": "MLP, layers 64 → 32", "mlp_bag10_64x32_a1": "10 bagged MLPs, 64 → 32",
        "torch_ple16_mlp128x64_k5": "PyTorch net, 16-bin encoding, 128 → 64, 5-net ensemble",
        "torch_ple32_mlp256x128_k5_drop0.3": "PyTorch net, 32-bin encoding, 256 → 128, dropout 0.3, 5-net ensemble",
        "torch_quantile_mlp128x64_k5": "PyTorch net, rank-transformed inputs, 128 → 64, 5-net ensemble",
        "torch_ple16_linear_k5": "PyTorch net, 16-bin encoding, no hidden layer, 5-net ensemble",
        "lgbm_monotone_rpm_fuel_oiltemp": "LightGBM with monotone constraints",
        "lgbm_stumps_lr0.02_n1500": "LightGBM stumps, 1,500 trees",
        "lgbm_linear_tree": "LightGBM with linear models in leaves",
        "lgbm_lambdarank_like_xentropy": "LightGBM, cross-entropy objective",
        "catboost_tuned_seedavg5": "CatBoost tuned, 5 seeds averaged",
        "ref_catboost_tuned": "CatBoost, team's tuned settings (re-run)",
        "ref_lightgbm_tuned": "LightGBM, team's tuned settings (re-run)",
        "stack_cb_spline": "Stack: CatBoost + spline LR",
        "stack_cb_spline_ebm_et_kde": "Stack: CatBoost + spline + EBM + extra trees + KDE NB",
        "lr_linear": "Logistic regression (straight lines)",
    }
    if n in fixed: return fixed[n]
    n2 = n.replace("[all]", "[raw + 16 dataset features]").replace("[raw+own]", "[raw + 4 own features]")
    n2 = n2.replace("[all, corr-dropped]", "[near-duplicates dropped]").replace("[all, top-12]", "[top 12 features]").replace("[all, top-8]", "[top 8 features]")
    return n2.replace("(params from #13)", "with #13 settings")

lb = [{**r, "desc": desc(r["name"])} for r in d["leaderboard"]]
curves = json.loads((HERE / "model_curves.json").read_text())
curves["units"] = {x["name"]: x["unit"] for x in d["sensors"]}
out = {"curves": curves, "leaderboard": lb, "test": [{k: t[k] for k in ("name", "auc", "lo", "hi", "brier", "by")} for t in d["test"]]}
page = (HERE / "models_template.html").read_text().replace("__DATA__", json.dumps(out))
import sys
art = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if art: art.write_text(page)
head = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
title_end = page.index("</style>") + len("</style>")
(HERE.parent / "reports" / "smartops_models.html").write_text(head + page[:title_end] + "\n</head>\n<body>\n" + page[title_end:] + "\n</body>\n</html>\n")
print("ok", len(page))
