from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

from .config import DEFAULT_TOP_K


def _get_feature_names(pre: ColumnTransformer, input_feature_names: List[str]) -> List[str]:
    try:
        return list(pre.get_feature_names_out(input_features=input_feature_names))
    except Exception:
        try:
            return list(pre.get_feature_names_out())
        except Exception:
            return input_feature_names


def _extract_estimator(pipeline) -> BaseEstimator:
    return pipeline.named_steps["clf"]


def _is_lgbm(clf: BaseEstimator) -> bool:
    return hasattr(clf, "booster_") or clf.__class__.__name__.startswith("LGBM")


def _require_lgbm(clf: BaseEstimator) -> None:
    if not _is_lgbm(clf):
        raise ValueError("SHAP demo is LightGBM-only. Given model is not LightGBM.")


def _resolve_feature_names(pipeline, pre: ColumnTransformer, X: pd.DataFrame, n_features: int) -> List[str]:
    cached = getattr(pipeline, "_shap_cache", {}).get("feature_names") if hasattr(pipeline, "_shap_cache") else None
    if cached and len(cached) == n_features:
        return cached
    names = _get_feature_names(pre, list(X.columns))
    if len(names) != n_features:
        names = [f"feature_{i}" for i in range(n_features)]
    return names


def _split_contributions(contribs: np.ndarray, pred_class: int, n_features: int) -> Tuple[np.ndarray, float]:
    """Extract (values, base_value) from LightGBM pred_contrib output.

    Handles the three output shapes LightGBM may emit:
      (1, n_features + 1), (1, n_classes * (n_features + 1)), (1, n_classes, n_features + 1).
    """
    if contribs.ndim == 2 and contribs.shape[1] == n_features + 1:
        return contribs[0][:-1], float(contribs[0][-1])
    if contribs.ndim == 2 and contribs.shape[1] % (n_features + 1) == 0:
        start = pred_class * (n_features + 1)
        return contribs[0][start : start + n_features], float(contribs[0][start + n_features])
    if contribs.ndim == 3 and contribs.shape[-1] == n_features + 1:
        return contribs[0, pred_class, :-1], float(contribs[0, pred_class, -1])
    raise RuntimeError(f"Unexpected pred_contrib shape: {contribs.shape}, n_features={n_features}")


def _compute_contributions(pipeline, X: pd.DataFrame, index: int):
    clf = _extract_estimator(pipeline)
    _require_lgbm(clf)
    pre = pipeline.named_steps["pre"]

    x_row = X.iloc[[index]]
    x_row_t = pre.transform(x_row)

    booster = getattr(clf, "booster_", None)
    if booster is None:
        raise RuntimeError("LightGBM booster not found on classifier.")

    probs = clf.predict_proba(x_row_t)[0]
    pred_class = int(np.argmax(probs))

    contribs = booster.predict(x_row_t, pred_contrib=True)
    if hasattr(contribs, "toarray"):
        contribs = contribs.toarray()
    contribs = np.asarray(contribs)

    n_features = x_row_t.shape[1]
    values, base = _split_contributions(contribs, pred_class, n_features)
    feature_names = _resolve_feature_names(pipeline, pre, X, n_features)

    x_dense = x_row_t.toarray()[0] if hasattr(x_row_t, "toarray") else np.asarray(x_row_t)[0]

    return {
        "values": values,
        "base": base,
        "feature_names": feature_names,
        "data": x_dense,
        "probs": probs,
        "pred_class": pred_class,
    }


def prime_shap_cache(pipeline, X: pd.DataFrame) -> None:
    """Cache transformed feature names so per-row explanations avoid recomputation."""
    clf = _extract_estimator(pipeline)
    _require_lgbm(clf)
    pre = pipeline.named_steps["pre"]
    pipeline._shap_cache = {"feature_names": _get_feature_names(pre, list(X.columns))}  # type: ignore[attr-defined]


def top_contributors_for_instance(
    pipeline, X: pd.DataFrame, index: int, top_k: int = DEFAULT_TOP_K
) -> Dict[str, Any]:
    info = _compute_contributions(pipeline, X, index)
    values = info["values"]
    order = np.argsort(np.abs(values))[::-1][:top_k]
    top = [(info["feature_names"][i], float(values[i])) for i in order]
    return {
        "index": int(index),
        "pred_class": info["pred_class"] + 1,
        "top_features": top,
        "probs": info["probs"],
    }


def waterfall_figure_for_instance(
    pipeline, X: pd.DataFrame, index: int, top_k: int = DEFAULT_TOP_K
):
    info = _compute_contributions(pipeline, X, index)
    values, base = info["values"], info["base"]

    order = np.argsort(np.abs(values))[::-1][:top_k]
    exp = shap.Explanation(
        values=values[order],
        base_values=base,
        data=info["data"][order],
        feature_names=[info["feature_names"][i] for i in order],
    )

    plt.close("all")
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(exp, max_display=top_k, show=False)
    return plt.gcf()


def build_explainer(pipeline, X_sample: pd.DataFrame):
    """Legacy helper kept for notebook compatibility."""
    clf = _extract_estimator(pipeline)
    pre = pipeline.named_steps["pre"]
    X_trans = pre.fit_transform(X_sample)
    feature_names = _get_feature_names(pre, list(X_sample.columns))

    if _is_lgbm(clf):
        explainer = shap.TreeExplainer(
            clf, feature_perturbation="tree_path_dependent", model_output="probability"
        )
    else:
        background = shap.sample(X_trans, 100, random_state=0)
        explainer = shap.KernelExplainer(lambda data: clf.predict_proba(data), background)
    return explainer, feature_names, X_trans
