from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import streamlit as st

from ..config import DATA_DIR, MODEL_PATH
from ..data import load_test
from ..persist import load_pipeline
from ..shap_utils import (
    _extract_estimator,
    _is_lgbm,
    compute_dataset_shap,
    prime_shap_cache,
)


class ResourceLoadError(RuntimeError):
    """Raised when model or data cannot be loaded."""


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


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise ResourceLoadError(
            f"{label} 파일을 찾을 수 없습니다: {path}. "
            f"Kaggle Prudential 데이터를 `data/raw/`에 배치했는지 확인하세요."
        )


@st.cache_resource(show_spinner="Loading model and data...")
def load_resources() -> Resources:
    _require_file(DATA_DIR / "test.csv", "test.csv")
    _require_file(Path(str(MODEL_PATH)), "final_pipe.joblib")

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


@st.cache_resource(show_spinner="Computing dataset-level SHAP values...")
def load_dataset_shap(_pipeline, X: pd.DataFrame, max_samples: int = 200) -> Optional[dict[str, Any]]:
    clf = _extract_estimator(_pipeline)
    if not _is_lgbm(clf):
        return None
    try:
        return compute_dataset_shap(_pipeline, X, max_samples=max_samples)
    except Exception:
        return None


def get_model_info(pipeline) -> ModelInfo:
    clf = _extract_estimator(pipeline)
    name = clf.__class__.__name__
    n_features = int(getattr(clf, "n_features_in_", 0)) or len(getattr(pipeline, "feature_names_in_", []))
    n_classes = int(getattr(clf, "n_classes_", 0))
    return ModelInfo(name=name, n_features=n_features, n_classes=n_classes, is_lgbm=_is_lgbm(clf))
