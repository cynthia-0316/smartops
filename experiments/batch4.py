import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from harness import run, X
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, StandardScaler, TargetEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.linear_model import LogisticRegression
COLS = list(X.columns)
S = lambda k, d=3: SplineTransformer(n_knots=k, degree=d, knots="quantile", extrapolation="linear")

def per_feature(knots, C=0.03, extra=None):
    tr = [(c, S(knots[c]), [c]) for c in COLS]
    if extra: tr += extra
    return make_pipeline(ColumnTransformer(tr), StandardScaler(), LogisticRegression(C=C, max_iter=5000))

base = {c: 6 for c in COLS}
for rk in (8, 10, 15):
    run(f"spl_rpm{rk}_others6", lambda rk=rk: per_feature({**base, "Engine RPM": rk}))
run("spl_rpm10_others4", lambda: per_feature({**{c: 4 for c in COLS}, "Engine RPM": 10}))
run("spl_rpm10_fuel8_others5", lambda: per_feature({**{c: 5 for c in COLS}, "Engine RPM": 10, "Fuel Pressure": 8}))
for deg in (1, 2):
    run(f"spl_k6_degree{deg}", lambda deg=deg: make_pipeline(S(6, deg), StandardScaler(), LogisticRegression(C=0.03, max_iter=5000)))
# RPM as a category (cross-fitted target encoding) on top of smooth curves
run("spl_k6_plus_rpm_target_enc", lambda: per_feature(base, extra=[("rpm_te", make_pipeline(FunctionTransformer(lambda a: np.asarray(a).astype(int).astype(str)), TargetEncoder(random_state=0)), ["Engine RPM"])]))
# one explicit interaction: tensor product of RPM and fuel pressure splines
class Tensor:
    pass
from sklearn.base import BaseEstimator, TransformerMixin
class PairSpline(BaseEstimator, TransformerMixin):
    def __init__(self, k=4): self.k = k
    def fit(self, Xp, y=None):
        Xp = np.asarray(Xp); self.a_ = S(self.k).fit(Xp[:, [0]]); self.b_ = S(self.k).fit(Xp[:, [1]]); return self
    def transform(self, Xp):
        Xp = np.asarray(Xp); A = self.a_.transform(Xp[:, [0]]); B = self.b_.transform(Xp[:, [1]])
        return np.einsum("ni,nj->nij", A, B).reshape(len(Xp), -1)
for pair in [("Engine RPM", "Fuel Pressure"), ("Engine RPM", "Lub Oil Temp")]:
    run(f"spl_k6_plus_tensor_{pair[0][:6]}_{pair[1][:6]}".replace(" ", ""), lambda pair=pair: per_feature(base, C=0.03, extra=[("pair", PairSpline(4), list(pair))]))
try:
    from pygam import LogisticGAM, s
    class GAM(BaseEstimator):
        def __init__(self, n_splines=12): self.n_splines = n_splines
        def fit(self, Xa, y):
            Xa = np.asarray(Xa); terms = s(0, n_splines=self.n_splines)
            for j in range(1, Xa.shape[1]): terms += s(j, n_splines=self.n_splines)
            self.g_ = LogisticGAM(terms).gridsearch(Xa, np.asarray(y), lam=np.logspace(-1, 3, 9), progress=False); return self
        def predict_proba(self, Xa):
            p = self.g_.predict_mu(np.asarray(Xa)); return np.c_[1 - p, p]
    run("pygam_logistic_gridsearch_lam", lambda: GAM(12))
except Exception as e:
    print("pygam failed:", repr(e)[:200])
