import warnings; warnings.filterwarnings("ignore")
from harness import run
from models import spline_lr, KDENaiveBayes, BP
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from interpret.glassbox import ExplainableBoostingClassifier

cb = BP("best_params_catboost.json"); lg = BP("best_params_lightgbm.json")
run("ref_catboost_tuned", lambda: CatBoostClassifier(**cb, random_seed=42, verbose=0, thread_count=1), note="params tuned on all dev in #13 (slightly optimistic)")
run("ref_lightgbm_tuned", lambda: LGBMClassifier(**lg, subsample_freq=1, random_state=42, n_jobs=1, verbose=-1))
run("lr_linear", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))
for k in (4, 6, 8, 12):
    for C in (0.03, 0.3, 3):
        run(f"lr_spline_k{k}_C{C}", lambda k=k, C=C: spline_lr(n_knots=k, C=C))
run("gnb", lambda: GaussianNB())
run("gnb_quantile_normal", lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=0), GaussianNB()))
run("kde_nb", lambda: KDENaiveBayes())
run("kde_nb_bw0.3", lambda: KDENaiveBayes(bw=0.3))
run("ebm_default", lambda: ExplainableBoostingClassifier(random_state=42, n_jobs=1))
run("ebm_no_interactions", lambda: ExplainableBoostingClassifier(interactions=0, random_state=42, n_jobs=1))
for k in (100, 300, 600):
    run(f"knn_quantile_k{k}", lambda k=k: make_pipeline(QuantileTransformer(random_state=0), KNeighborsClassifier(k, weights="distance")))
run("rf_leaf50", lambda: RandomForestClassifier(500, min_samples_leaf=50, max_features=0.5, random_state=42, n_jobs=1))
run("et_leaf30", lambda: ExtraTreesClassifier(500, min_samples_leaf=30, max_features=0.7, random_state=42, n_jobs=1))
run("mlp_64_32", lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=0),
                                       MLPClassifier((64, 32), alpha=1e-2, early_stopping=True, max_iter=500, random_state=42)))
