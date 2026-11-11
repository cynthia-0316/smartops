import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import loguniform, randint
from harness import run
from models import spline_lr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from catboost import CatBoostClassifier

INNER = StratifiedKFold(3, shuffle=True, random_state=42)
spl_grid = lambda: GridSearchCV(spline_lr(), {"splinetransformer__n_knots": [4, 6, 8, 12], "logisticregression__C": [0.03, 0.3, 3]},
                                cv=INNER, scoring="roc_auc", n_jobs=1)
run("NESTED_spline_lr_grid", spl_grid, note="nested: grid over knots x C inside each outer fold")

cb_space = {"depth": randint(1, 5), "learning_rate": loguniform(0.02, 0.2), "iterations": randint(100, 800), "l2_leaf_reg": loguniform(1, 30)}
cb_search = lambda: RandomizedSearchCV(CatBoostClassifier(random_seed=42, verbose=0, thread_count=1), cb_space, n_iter=15, cv=INNER,
                                       scoring="roc_auc", random_state=42, n_jobs=1)
run("NESTED_catboost_search", cb_search, note="nested: 15-setting search inside each outer fold")
stack = lambda: StackingClassifier([("cb", cb_search()), ("spl", spl_grid())], final_estimator=LogisticRegression(C=1.0),
                                   cv=StratifiedKFold(5, shuffle=True, random_state=1), n_jobs=1)
run("NESTED_stack_catboost_spline", stack, note="nested: both base models tuned inside each outer fold")
