import warnings; warnings.filterwarnings("ignore")
import numpy as np
from harness import run
from models import spline_lr, BP, KDENaiveBayes
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from interpret.glassbox import ExplainableBoostingClassifier

cb = BP("best_params_catboost.json")
CB = lambda s=42: CatBoostClassifier(**cb, random_seed=s, verbose=0, thread_count=1)

class Winsor(BaseEstimator):
    def fit(self, X, y=None): self.lo_, self.hi_ = np.percentile(X, 0.5, 0), np.percentile(X, 99.5, 0); return self
    def transform(self, X): return np.clip(np.asarray(X, float), self.lo_, self.hi_)
run("lr_spline_k6_C0.03_winsor", lambda: make_pipeline(Winsor(), spline_lr(6, C=0.03)))
run("lr_spline_k6_C0.03_quantile_uniform", lambda: spline_lr(6, C=0.03, quantile_first=True, knots="uniform"))

class NoiseFilter(BaseEstimator, ClassifierMixin):
    """Drop training rows whose out-of-fold probability for their own label is below q, then refit."""
    def __init__(self, base, q=0.2): self.base, self.q = base, q
    def fit(self, X, y):
        y = np.asarray(y); p = cross_val_predict(clone(self.base), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        own = np.where(y == 1, p, 1 - p); keep = own >= self.q
        self.m_ = clone(self.base).fit(X[keep] if not hasattr(X, "iloc") else X.iloc[keep], y[keep]); self.classes_ = self.m_.classes_; return self
    def predict_proba(self, X): return self.m_.predict_proba(X)
for q in (0.15, 0.3):
    run(f"noisefilter_q{q}_spline_lr", lambda q=q: NoiseFilter(spline_lr(6, C=0.03), q))

run("lgbm_lambdarank_like_xentropy", lambda: LGBMClassifier(objective="cross_entropy", num_leaves=4, learning_rate=0.02, n_estimators=400, min_child_samples=24, subsample=0.77, subsample_freq=1, colsample_bytree=0.84, random_state=42, n_jobs=1, verbose=-1))

stack = lambda: StackingClassifier(
    [("cb", CB()), ("spl", spline_lr(6, C=0.03)), ("ebm", ExplainableBoostingClassifier(random_state=42, n_jobs=1)),
     ("et", ExtraTreesClassifier(300, min_samples_leaf=30, max_features=0.7, random_state=42, n_jobs=1)), ("kde", KDENaiveBayes(bw=0.3))],
    final_estimator=LogisticRegression(C=1.0), cv=5, stack_method="predict_proba", n_jobs=1)
run("stack_cb_spline_ebm_et_kde", stack)
