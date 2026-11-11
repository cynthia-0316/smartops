import warnings; warnings.filterwarnings("ignore")
import numpy as np
from harness import run, X, corrected_ttest
from sklearn.base import BaseEstimator
from pygam import LogisticGAM, s, te

class GAM(BaseEstimator):
    def __init__(self, n_splines=12, lams=np.logspace(-1, 3, 9), rpm_constraint=None, inter=None, spline_order=3):
        self.n_splines, self.lams, self.rpm_constraint, self.inter, self.spline_order = n_splines, lams, rpm_constraint, inter, spline_order
    def fit(self, Xa, y):
        Xa = np.asarray(Xa)
        terms = s(0, n_splines=self.n_splines, constraints=self.rpm_constraint, spline_order=self.spline_order)
        for j in range(1, Xa.shape[1]): terms += s(j, n_splines=self.n_splines, spline_order=self.spline_order)
        if self.inter: terms += te(*self.inter, n_splines=5)
        self.g_ = LogisticGAM(terms).gridsearch(Xa, np.asarray(y), lam=self.lams, progress=False); return self
    def predict_proba(self, Xa):
        p = self.g_.predict_mu(np.asarray(Xa)); return np.c_[1 - p, p]

for ns in (8, 20, 30):
    run(f"pygam_ns{ns}", lambda ns=ns: GAM(ns))
run("pygam_ns12_rpm_monotone_dec", lambda: GAM(12, rpm_constraint="monotonic_dec"))
run("pygam_ns12_quadratic", lambda: GAM(12, spline_order=2))
run("pygam_ns12_te_rpm_oiltemp", lambda: GAM(12, inter=(0, 4)))
a = lambda n: np.load(f"preds/{n}__aucs.npy")
for n in ("pygam_logistic_gridsearch_lam", "pygam_ns20"):
    for ref in ("lr_spline_k6_C0.03", "ref_catboost_tuned", "NESTED_stack_catboost_spline"):
        d, p = corrected_ttest(a(n), a(ref)); print(f"{n} vs {ref}: {d:+.4f} p={p:.3f}")
