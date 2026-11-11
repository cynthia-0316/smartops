"""Model factories used in the experiments."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer, QuantileTransformer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
BP = lambda f: json.loads((ROOT / "results" / f).read_text())


def spline_lr(n_knots=6, degree=3, C=0.3, knots="quantile", quantile_first=False):
    steps = []
    if quantile_first:
        steps.append(QuantileTransformer(n_quantiles=1000, output_distribution="uniform", random_state=0))
    steps += [SplineTransformer(n_knots=n_knots, degree=degree, knots=knots, extrapolation="linear"),
              StandardScaler(), LogisticRegression(C=C, max_iter=5000)]
    return make_pipeline(*steps)


class KDENaiveBayes(BaseEstimator, ClassifierMixin):
    """Naive Bayes with a 1-D Gaussian KDE per feature and class (on quantile-normalised inputs)."""
    def __init__(self, bw=None):
        self.bw = bw

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        self.qt_ = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0).fit(X)
        Z = self.qt_.transform(X)
        self.classes_ = np.array([0, 1])
        self.prior_ = np.array([np.mean(y == 0), np.mean(y == 1)])
        self.kdes_ = [[gaussian_kde(Z[y == c, j], bw_method=self.bw) for j in range(Z.shape[1])] for c in (0, 1)]
        return self

    def predict_proba(self, X):
        Z = self.qt_.transform(np.asarray(X, float))
        ll = np.zeros((Z.shape[0], 2))
        for c in (0, 1):
            ll[:, c] = np.log(self.prior_[c]) + sum(np.log(self.kdes_[c][j](Z[:, j]) + 1e-300) for j in range(Z.shape[1]))
        ll -= ll.max(1, keepdims=True)
        p = np.exp(ll); p /= p.sum(1, keepdims=True)
        return p

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
