from typing import Dict

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .config import RANDOM_STATE
from .preprocess import build_preprocessor_from_df


def make_pipelines(df_sample) -> Dict[str, Pipeline]:
    """Create baseline pipelines for the binary underwriting task.

    Target: 1 = reject (Response == 1), 0 = approve (Response ∈ {2..8}).
    Use `src.data.binarize_target(df)` before fitting these pipelines.
    """
    pre_linear = build_preprocessor_from_df(df_sample, for_linear=True)
    pre_tree = build_preprocessor_from_df(df_sample, for_linear=False)

    logit = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    svm = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    lgbm = LGBMClassifier(
        objective="binary",
        random_state=RANDOM_STATE,
        n_estimators=1000,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    return {
        "logistic_regression": Pipeline([("pre", pre_linear), ("clf", logit)]),
        "random_forest": Pipeline([("pre", pre_tree), ("clf", rf)]),
        "svm_rbf": Pipeline([("pre", pre_linear), ("clf", svm)]),
        "lightgbm": Pipeline([("pre", pre_tree), ("clf", lgbm)]),
    }
