from __future__ import annotations
import os
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from .shap_utils import _extract_estimator  # reuse to validate


def save_pipeline(pipeline: Pipeline, path: str) -> str:
    """Save a fitted sklearn Pipeline to disk.

    Ensures parent directory exists. Returns the absolute path saved to.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Expected sklearn.pipeline.Pipeline")
    # basic validation: must have 'pre' and 'clf'
    if not hasattr(pipeline, "named_steps") or not {"pre", "clf"}.issubset(set(pipeline.named_steps.keys())):
        raise ValueError("Pipeline must contain 'pre' and 'clf' steps")

    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    joblib.dump(pipeline, abs_path)
    return abs_path


essential_keys = {"pre", "clf"}

def load_pipeline(path: str) -> Pipeline:
    """Load a fitted sklearn Pipeline from disk and perform basic integrity checks."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Model file not found: {abs_path}")
    pipe = joblib.load(abs_path)
    if not isinstance(pipe, Pipeline):
        raise TypeError("Loaded object is not an sklearn Pipeline")
    if not hasattr(pipe, "named_steps") or not essential_keys.issubset(set(pipe.named_steps.keys())):
        raise ValueError("Loaded Pipeline does not contain required 'pre' and 'clf' steps")
    # touch estimator to ensure attribute access works
    _ = _extract_estimator(pipe)
    return pipe


def load_and_prime(path: str, X_sample: Any) -> Pipeline:
    """Load a pipeline and prime SHAP cache for LightGBM using a sample of X.

    X_sample can be the full X or a small slice with the same columns.
    """
    from .shap_utils import prime_shap_cache

    pipe = load_pipeline(path)
    prime_shap_cache(pipe, X_sample)
    return pipe
