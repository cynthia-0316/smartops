import warnings; warnings.filterwarnings("ignore")
from harness import run, X
from models import BP
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

QN = lambda: QuantileTransformer(output_distribution="normal", random_state=0)
for hid in [(32,), (128, 64), (256, 128, 64)]:
    for a in (1e-1, 1.0):
        run(f"mlp_{'x'.join(map(str,hid))}_a{a}", lambda hid=hid, a=a: make_pipeline(QN(), MLPClassifier(hid, alpha=a, early_stopping=True, n_iter_no_change=20, max_iter=1000, random_state=42)))
run("mlp_bag10_64x32_a1", lambda: make_pipeline(QN(), BaggingClassifier(MLPClassifier((64, 32), alpha=1.0, early_stopping=True, n_iter_no_change=20, max_iter=1000), n_estimators=10, random_state=42)))
cb = BP("best_params_catboost.json")
for d in (1, 2, 3, 4):
    run(f"catboost_depth{d}_lr0.03_it1000", lambda d=d: CatBoostClassifier(depth=d, learning_rate=0.03, iterations=1000, l2_leaf_reg=10, random_seed=42, verbose=0, thread_count=1))
# monotone constraints in the direction of the univariate effect (RPM -, fuel +, oil temp -)
mono = [-1, 0, 1, 0, -1, 0]
run("lgbm_monotone_rpm_fuel_oiltemp", lambda: LGBMClassifier(num_leaves=4, learning_rate=0.02, n_estimators=500, min_child_samples=30, subsample=0.8, subsample_freq=1, monotone_constraints=mono, random_state=42, n_jobs=1, verbose=-1))
run("lgbm_stumps_lr0.02_n1500", lambda: LGBMClassifier(num_leaves=2, learning_rate=0.02, n_estimators=1500, min_child_samples=30, subsample=0.8, subsample_freq=1, random_state=42, n_jobs=1, verbose=-1))
run("lgbm_linear_tree", lambda: LGBMClassifier(num_leaves=4, learning_rate=0.02, n_estimators=400, linear_tree=True, linear_lambda=1.0, min_child_samples=50, random_state=42, n_jobs=1, verbose=-1))
# catboost with the tuned params, many seeds averaged
class SeedAvg:
    def __init__(self, make, seeds): self.make, self.seeds = make, seeds
    def fit(self, X, y): self.ms = [self.make(s).fit(X, y) for s in self.seeds]; return self
    def predict_proba(self, X):
        import numpy as np; return np.mean([m.predict_proba(X) for m in self.ms], 0)
run("catboost_tuned_seedavg5", lambda: SeedAvg(lambda s: CatBoostClassifier(**cb, random_seed=s, verbose=0, thread_count=1), range(5)))
