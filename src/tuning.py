from __future__ import annotations
from typing import Optional
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier

from .config import RANDOM_STATE
from .preprocess import build_preprocessor_from_df
from .metrics import qwk_scorer
from sklearn.pipeline import Pipeline


def tune_lightgbm(df_sample, X, y, n_trials: int = 30, n_splits: int = 5, random_state: int = RANDOM_STATE) -> tuple[LGBMClassifier, dict]:
    pre_tree = build_preprocessor_from_df(df_sample, for_linear=False)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multiclass",
            "num_class": 8,
            "random_state": random_state,
            "n_estimators": trial.suggest_int("n_estimators", 400, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = LGBMClassifier(**params)
        pipe = Pipeline([("pre", pre_tree), ("clf", model)])
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe, X, y, scoring=qwk_scorer, cv=cv, n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({"objective": "multiclass", "num_class": 8, "random_state": random_state})
    best_model = LGBMClassifier(**best_params)
    return best_model, best_params
