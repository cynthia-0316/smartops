"""Modern tabular neural nets: piecewise-linear encoding (PLE) + MLP, deep ensembles, early stopping on an inner split."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from harness import run

class PLE:
    def fit(self, X, n_bins=16):
        self.edges = [np.unique(np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1))) for j in range(X.shape[1])]; return self
    def transform(self, X):
        out = []
        for j, e in enumerate(self.edges):
            lo, hi = e[:-1], e[1:]
            v = np.clip((X[:, [j]] - lo) / (hi - lo), 0, 1)
            out.append(v)
        return np.hstack(out).astype(np.float32)

class TorchTab(BaseEstimator):
    def __init__(self, enc="ple", hidden=(128, 64), dropout=0.2, wd=1e-3, lr=1e-3, k=5, n_bins=16, epochs=200, patience=20, seed=42):
        self.enc, self.hidden, self.dropout, self.wd, self.lr, self.k, self.n_bins, self.epochs, self.patience, self.seed = enc, hidden, dropout, wd, lr, k, n_bins, epochs, patience, seed
    def _prep_fit(self, X):
        if self.enc == "ple": self.t_ = PLE().fit(X, self.n_bins)
        else: self.t_ = QuantileTransformer(output_distribution="normal", random_state=0).fit(X)
        return self._prep(X)
    def _prep(self, X): return np.asarray(self.t_.transform(X), np.float32)
    def _net(self, d):
        layers, prev = [], d
        for h in self.hidden: layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]; prev = h
        return nn.Sequential(*layers, nn.Linear(prev, 1))
    def fit(self, X, y):
        torch.set_num_threads(1); X = np.asarray(X, float); y = np.asarray(y, np.float32)
        Z = self._prep_fit(X); self.nets_ = []
        for m in range(self.k):
            torch.manual_seed(self.seed + m)
            tr, va = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=self.seed + m)
            net = self._net(Z.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            Zt, yt = torch.tensor(Z[tr]), torch.tensor(y[tr]); Zv = torch.tensor(Z[va])
            best, best_state, bad = -1, None, 0
            for ep in range(self.epochs):
                net.train(); perm = torch.randperm(len(tr))
                for i in range(0, len(tr), 256):
                    b = perm[i:i + 256]; opt.zero_grad()
                    loss = nn.functional.binary_cross_entropy_with_logits(net(Zt[b]).squeeze(1), yt[b]); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad(): auc = roc_auc_score(y[va], net(Zv).squeeze(1).numpy())
                if auc > best + 1e-4: best, bad, best_state = auc, 0, {k_: v.clone() for k_, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience: break
            net.load_state_dict(best_state); self.nets_.append(net)
        return self
    def predict_proba(self, X):
        Z = torch.tensor(self._prep(np.asarray(X, float)))
        with torch.no_grad(): p = np.mean([torch.sigmoid(n(Z)).squeeze(1).numpy() for n in self.nets_], 0)
        return np.c_[1 - p, p]

run("torch_ple16_mlp128x64_k5", lambda: TorchTab())
run("torch_ple32_mlp256x128_k5_drop0.3", lambda: TorchTab(n_bins=32, hidden=(256, 128), dropout=0.3))
run("torch_quantile_mlp128x64_k5", lambda: TorchTab(enc="quantile"))
run("torch_ple16_linear_k5", lambda: TorchTab(hidden=(), dropout=0.0, wd=1e-4, lr=3e-3))
