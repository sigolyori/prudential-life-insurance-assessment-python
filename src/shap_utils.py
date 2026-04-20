from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

from .config import DEFAULT_TOP_K, REJECT_CLASS


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
    clf = _extract_estimator(pipeline)
    classes = getattr(clf, "classes_", None)
    pred_label = int(classes[info["pred_class"]]) if classes is not None else info["pred_class"]
    return {
        "index": int(index),
        "pred_class": pred_label,
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


def _reject_class_index(clf: BaseEstimator) -> int:
    classes = list(getattr(clf, "classes_", []))
    if REJECT_CLASS in classes:
        return classes.index(REJECT_CLASS)
    return 0


def compute_dataset_shap(
    pipeline, X: pd.DataFrame, max_samples: int = 200
) -> Dict[str, Any]:
    """Compute SHAP values for a sampled subset (LightGBM-only).

    Returns per-sample SHAP values aimed at the reject class so the resulting
    plots emphasize risk drivers. Works for binary (classes=[0,1]) and the
    legacy 8-class models alike.
    """
    clf = _extract_estimator(pipeline)
    _require_lgbm(clf)
    pre = pipeline.named_steps["pre"]

    X_sub = X.head(max_samples) if len(X) > max_samples else X
    X_trans = pre.transform(X_sub)
    if hasattr(X_trans, "toarray"):
        X_trans_dense = X_trans.toarray()
    else:
        X_trans_dense = np.asarray(X_trans)

    booster = getattr(clf, "booster_", None)
    if booster is None:
        raise RuntimeError("LightGBM booster not found on classifier.")

    contribs = booster.predict(X_trans_dense, pred_contrib=True)
    if hasattr(contribs, "toarray"):
        contribs = contribs.toarray()
    contribs = np.asarray(contribs)

    n_samples, n_features = X_trans_dense.shape
    reject_idx = _reject_class_index(clf)

    if contribs.ndim == 2 and contribs.shape[1] == n_features + 1:
        values = contribs[:, :-1]
    elif contribs.ndim == 2 and contribs.shape[1] % (n_features + 1) == 0:
        start = reject_idx * (n_features + 1)
        values = contribs[:, start : start + n_features]
    elif contribs.ndim == 3 and contribs.shape[-1] == n_features + 1:
        values = contribs[:, reject_idx, :-1]
    else:
        raise RuntimeError(f"Unexpected pred_contrib shape: {contribs.shape}")

    feature_names = _resolve_feature_names(pipeline, pre, X_sub, n_features)
    return {
        "shap_values": values,
        "X_transformed": X_trans_dense,
        "feature_names": feature_names,
        "n_samples": n_samples,
    }


def summary_figure(bundle: Dict[str, Any], max_display: int = 15):
    plt.close("all")
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        bundle["shap_values"],
        bundle["X_transformed"],
        feature_names=bundle["feature_names"],
        max_display=max_display,
        show=False,
    )
    return plt.gcf()


def feature_importance_figure(bundle: Dict[str, Any], max_display: int = 15):
    mean_abs = np.mean(np.abs(bundle["shap_values"]), axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    names = [bundle["feature_names"][i] for i in order]
    values = mean_abs[order]

    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(order))[::-1], values, color="#4e79a7")
    ax.set_yticks(range(len(order))[::-1])
    ax.set_yticklabels(names)
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("Feature importance (reject class)")
    fig.tight_layout()
    return fig


def dependency_figure(bundle: Dict[str, Any], feature: str):
    feature_names = bundle["feature_names"]
    if feature not in feature_names:
        raise ValueError(f"Feature '{feature}' not found among SHAP features.")
    plt.close("all")
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        feature,
        bundle["shap_values"],
        bundle["X_transformed"],
        feature_names=feature_names,
        show=False,
    )
    return plt.gcf()


def ranked_feature_names(bundle: Dict[str, Any], limit: int = 20) -> List[str]:
    mean_abs = np.mean(np.abs(bundle["shap_values"]), axis=0)
    order = np.argsort(mean_abs)[::-1][:limit]
    return [bundle["feature_names"][i] for i in order]


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
