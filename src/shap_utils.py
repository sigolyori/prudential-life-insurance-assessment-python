from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import shap

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt


def _get_feature_names(pre: ColumnTransformer, input_feature_names: List[str]) -> List[str]:
    try:
        return list(pre.get_feature_names_out(input_features=input_feature_names))
    except Exception:
        # Fallback: return input names; OHE columns may be expanded without names
        return input_feature_names


def _extract_estimator(pipeline) -> BaseEstimator:
    return pipeline.named_steps["clf"]


def _is_lgbm(clf: BaseEstimator) -> bool:
    return hasattr(clf, "booster_") or clf.__class__.__name__.startswith("LGBM")


def prime_shap_cache(pipeline, X: pd.DataFrame) -> None:
    """Prepare and cache metadata for fast per-row explanations (LightGBM-only).

    We cache only feature names. For LightGBM, we use native pred_contrib=True
    which is faster than a SHAP TreeExplainer and requires no separate explainer.
    """
    clf = _extract_estimator(pipeline)
    if not _is_lgbm(clf):
        raise ValueError("SHAP demo is LightGBM-only. Given model is not LightGBM.")
    pre = pipeline.named_steps["pre"]
    feature_names = _get_feature_names(pre, list(X.columns))
    pipeline._shap_cache = {  # type: ignore[attr-defined]
        "feature_names": feature_names,
    }


def build_explainer(pipeline, X_sample: pd.DataFrame):
    clf = _extract_estimator(pipeline)
    pre = pipeline.named_steps["pre"]
    X_trans = pre.fit_transform(X_sample)

    if hasattr(clf, "booster_") or clf.__class__.__name__.startswith("LGBM"):
        # Keep helper returning a standard explainer tuple for compatibility
        explainer = shap.TreeExplainer(
            clf,
            feature_perturbation="tree_path_dependent",
            model_output="probability",
        )
        feature_names = _get_feature_names(pre, list(X_sample.columns))
        return explainer, feature_names, X_trans
    else:
        # KernelExplainer for arbitrary models (slow)
        background = shap.sample(X_trans, 100, random_state=0)
        f = lambda data: clf.predict_proba(data)
        explainer = shap.KernelExplainer(f, background)
        feature_names = _get_feature_names(pre, list(X_sample.columns))
        return explainer, feature_names, X_trans


def top_contributors_for_instance(pipeline, X: pd.DataFrame, index: int, top_k: int = 8) -> Dict[str, Any]:
    clf = _extract_estimator(pipeline)
    pre = pipeline.named_steps["pre"]

    x_row = X.iloc[[index]]
    x_row_t = pre.transform(x_row)
    # LightGBM-only fast path with caching
    if not _is_lgbm(clf):
        raise ValueError("SHAP demo is LightGBM-only. Given model is not LightGBM.")

    cache = getattr(pipeline, "_shap_cache", None)
    if not cache or "feature_names" not in cache:  # type: ignore[attr-defined]
        prime_shap_cache(pipeline, X)
        cache = getattr(pipeline, "_shap_cache", None)

    feature_names = cache["feature_names"]  # type: ignore[index]

    # Use LightGBM native contributions (fast)
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
    # Shapes observed:
    # 1) (n_samples, n_features+1)
    # 2) (n_samples, n_classes*(n_features+1))  [flattened]
    # 3) (n_samples, n_classes, n_features+1)
    if contribs.ndim == 2 and contribs.shape[1] == n_features + 1:
        values = contribs[0][:-1]
    elif contribs.ndim == 2 and contribs.shape[1] % (n_features + 1) == 0:
        n_classes_from_contrib = contribs.shape[1] // (n_features + 1)
        start = pred_class * (n_features + 1)
        end = start + n_features
        values = contribs[0][start:end]
    elif contribs.ndim == 3 and contribs.shape[-1] == n_features + 1:
        values = contribs[0, pred_class, :-1]
    else:
        raise RuntimeError(
            f"Unexpected pred_contrib shape: {contribs.shape}, n_features={n_features}"
        )

    order = np.argsort(np.abs(values))[::-1][:top_k]
    top = [(feature_names[i], float(values[i])) for i in order]
    return {
        "index": int(index),
        "pred_class": int(pred_class) + 1,  # dataset classes start at 1
        "top_features": top,
        "probs": probs,
    }


def waterfall_figure_for_instance(pipeline, X: pd.DataFrame, index: int, top_k: int = 8):
    """Create a SHAP waterfall matplotlib Figure for a single instance (LightGBM-only).

    Uses LightGBM native pred_contrib for speed. Selects top_k absolute contributors
    for readability and builds a shap.Explanation for plotting.
    """
    clf = _extract_estimator(pipeline)
    if not _is_lgbm(clf):
        raise ValueError("SHAP demo is LightGBM-only. Given model is not LightGBM.")
    pre = pipeline.named_steps["pre"]

    x_row = X.iloc[[index]]
    x_row_t = pre.transform(x_row)

    booster = getattr(clf, "booster_", None)
    if booster is None:
        raise RuntimeError("LightGBM booster not found on classifier.")

    # Contributions and base value per class
    contribs = booster.predict(x_row_t, pred_contrib=True)
    if hasattr(contribs, "toarray"):
        contribs = contribs.toarray()
    contribs = np.asarray(contribs)

    probs = clf.predict_proba(x_row_t)[0]
    pred_class = int(np.argmax(probs))

    n_features = x_row_t.shape[1]
    if contribs.ndim == 2 and contribs.shape[1] == n_features + 1:
        values = contribs[0][:-1]
        base = float(contribs[0][-1])
    elif contribs.ndim == 2 and contribs.shape[1] % (n_features + 1) == 0:
        start = pred_class * (n_features + 1)
        end = start + n_features
        values = contribs[0][start:end]
        base = float(contribs[0][start + n_features])
    elif contribs.ndim == 3 and contribs.shape[-1] == n_features + 1:
        values = contribs[0, pred_class, :-1]
        base = float(contribs[0, pred_class, -1])
    else:
        raise RuntimeError(
            f"Unexpected pred_contrib shape: {contribs.shape}, n_features={n_features}"
        )

    feature_names = getattr(pipeline, "_shap_cache", {}).get("feature_names")  # type: ignore[attr-defined]
    if not feature_names:
        feature_names = _get_feature_names(pre, list(X.columns))

    # Prepare top_k subset for readability
    order = np.argsort(np.abs(values))[::-1][:top_k]
    values_top = values[order]
    names_top = [feature_names[i] for i in order]

    # Get transformed data row and subset
    if hasattr(x_row_t, "toarray"):
        x_row_t_dense = x_row_t.toarray()[0]
    else:
        x_row_t_dense = np.asarray(x_row_t)[0]
    data_top = x_row_t_dense[order]

    # Build Explanation and plot
    exp = shap.Explanation(values=values_top, base_values=base, data=data_top, feature_names=names_top)
    plt.close('all')
    fig = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(exp, max_display=top_k, show=False)
    fig = plt.gcf()
    return fig
