from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
import streamlit as st

from ..config import MODEL_PATH
from ..data import load_test
from ..persist import load_pipeline
from ..shap_utils import prime_shap_cache, _extract_estimator, _is_lgbm


@dataclass
class Resources:
    X_test: pd.DataFrame
    pipeline: object
    feature_names: List[str]


@dataclass
class ModelInfo:
    name: str
    n_features: int
    n_classes: int
    is_lgbm: bool


@st.cache_resource(show_spinner="Loading model and data...")
def load_resources() -> Resources:
    X_test = load_test(drop_id=True)
    pipeline = load_pipeline(str(MODEL_PATH))

    if hasattr(pipeline, "feature_names_in_"):
        feature_names = list(pipeline.feature_names_in_)
    else:
        feature_names = list(X_test.columns)

    clf = _extract_estimator(pipeline)
    if _is_lgbm(clf):
        try:
            prime_shap_cache(pipeline, X_test)
        except Exception:
            pass

    return Resources(X_test=X_test, pipeline=pipeline, feature_names=feature_names)


def get_model_info(pipeline) -> ModelInfo:
    clf = _extract_estimator(pipeline)
    name = clf.__class__.__name__
    n_features = int(getattr(clf, "n_features_in_", 0)) or len(getattr(pipeline, "feature_names_in_", []))
    n_classes = int(getattr(clf, "n_classes_", 0))
    return ModelInfo(name=name, n_features=n_features, n_classes=n_classes, is_lgbm=_is_lgbm(clf))
