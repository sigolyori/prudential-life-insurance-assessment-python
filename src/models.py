from typing import Dict
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

from .config import RANDOM_STATE, TARGET_COL, ID_COL
from .preprocess import build_preprocessor_from_df


def make_pipelines(df_sample) -> Dict[str, Pipeline]:
    """Create baseline pipelines for multiple models (multiclass)."""
    # Preprocessors
    pre_linear = build_preprocessor_from_df(df_sample, for_linear=True)
    pre_tree = build_preprocessor_from_df(df_sample, for_linear=False)

    # Models
    logit = LogisticRegression(
        multi_class="multinomial",
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
        objective="multiclass",
        num_class=8,  # Prudential target classes 1~8
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
