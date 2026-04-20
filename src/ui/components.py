from __future__ import annotations

import io
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from ..config import DEFAULT_TOP_K, Language, decision_from_class
from ..genai import generate_underwriting_explanation
from ..shap_utils import (
    _extract_estimator,
    _is_lgbm,
    dependency_figure,
    feature_importance_figure,
    ranked_feature_names,
    summary_figure,
    top_contributors_for_instance,
    waterfall_figure_for_instance,
)
from .state import ModelInfo


_DECISION_COPY = {
    "ko": {
        "approve": ("인수 승인", "#2e7d32"),
        "reject": ("인수 거절", "#c62828"),
    },
    "en": {
        "approve": ("Approve", "#2e7d32"),
        "reject": ("Reject", "#c62828"),
    },
}


def render_sidebar(n_samples: int, model_info: ModelInfo) -> dict:
    """Render sidebar controls and return the current settings."""
    st.sidebar.header("Controls")
    sample_idx = st.sidebar.slider(
        "Test case index", min_value=0, max_value=max(n_samples - 1, 0), value=0
    )
    top_k = st.sidebar.slider(
        "SHAP top-K features", min_value=3, max_value=20, value=DEFAULT_TOP_K
    )
    language: Language = st.sidebar.radio(
        "Explanation language", options=("ko", "en"), horizontal=True, index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model")
    st.sidebar.markdown(
        f"- **Estimator:** `{model_info.name}`\n"
        f"- **Features:** {model_info.n_features}\n"
        f"- **Classes:** {model_info.n_classes}"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "LightGBM + SHAP + GenAI pipeline for Prudential underwriting."
    )
    return {"sample_idx": sample_idx, "top_k": top_k, "language": language}


def render_decision_badge(pred_class: int, language: Language) -> None:
    decision = decision_from_class(pred_class)
    label, color = _DECISION_COPY[language][decision]
    class_label = "예측 레이블" if language == "ko" else "Predicted label"
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:1rem;margin:0.25rem 0 1.25rem 0;">
          <span style="background:{color};color:white;padding:0.6rem 1.3rem;
                       border-radius:999px;font-weight:700;font-size:1.25rem;
                       letter-spacing:0.02em;">
            {label}
          </span>
          <span style="color:#555;font-size:0.95rem;">
            {class_label}: <b>{pred_class}</b>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_chart(
    pred_proba: np.ndarray, language: Language, class_labels: list | None = None
) -> None:
    header = "#### 클래스 확률" if language == "ko" else "#### Class probabilities"
    st.markdown(header)
    if class_labels is not None and len(class_labels) == len(pred_proba):
        if len(class_labels) == 2:
            approve_label = "승인(0)" if language == "ko" else "Approve (0)"
            reject_label = "거절(1)" if language == "ko" else "Reject (1)"
            labels_by_value = {0: approve_label, 1: reject_label}
            classes = [labels_by_value.get(int(c), f"Class {c}") for c in class_labels]
        else:
            classes = [f"Class {c}" for c in class_labels]
    else:
        classes = [f"Class {i + 1}" for i in range(len(pred_proba))]
    top = int(np.argmax(pred_proba))
    colors = ["#4e79a7" if i != top else "#f28e2b" for i in range(len(pred_proba))]

    fig = go.Figure(
        go.Bar(
            x=classes,
            y=[float(p) for p in pred_proba],
            marker_color=colors,
            text=[f"{p:.1%}" for p in pred_proba],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Probability", range=[0, min(1.0, float(pred_proba.max()) * 1.25)], tickformat=".0%"),
        xaxis=dict(title=""),
    )
    st.plotly_chart(fig, use_container_width=True)


def _fig_to_image(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def render_shap_panel(
    pipeline,
    sample: pd.DataFrame,
    dataset_bundle,
    top_k: int,
    language: Language,
) -> List[Tuple[str, float]]:
    header = "#### 피처 영향 분석" if language == "ko" else "#### Feature Impact"
    st.markdown(header)

    clf = _extract_estimator(pipeline)
    if not _is_lgbm(clf):
        msg = (
            "SHAP 분석은 LightGBM 모델에서만 지원됩니다."
            if language == "ko"
            else "SHAP analysis is only available for LightGBM models."
        )
        st.info(msg)
        return []

    tab_labels = (
        ["Waterfall", "Summary", "Feature Importance", "Dependency"]
        if language == "en"
        else ["Waterfall", "Summary", "중요도", "Dependency"]
    )
    tabs = st.tabs(tab_labels)

    top_features: List[Tuple[str, float]] = []

    with tabs[0]:
        try:
            with st.spinner("Computing local SHAP..."):
                contrib_info = top_contributors_for_instance(pipeline, sample, index=0, top_k=top_k)
                top_features = contrib_info.get("top_features", [])
                fig = waterfall_figure_for_instance(pipeline, sample, index=0, top_k=top_k)
            st.image(_fig_to_image(fig), use_container_width=True)
            st.caption(
                "파란 막대는 예측을 높이는 방향, 빨간 막대는 낮추는 방향의 피처 기여도입니다."
                if language == "ko"
                else "Blue bars push the prediction up; red bars push it down."
            )
        except Exception as e:
            st.warning(f"Waterfall failed: {e}")

    if dataset_bundle is None:
        msg = (
            "Summary / Importance / Dependency는 train.csv가 있을 때 활성화됩니다."
            if language == "ko"
            else "Summary / Importance / Dependency require train.csv to be available."
        )
        for tab in tabs[1:]:
            with tab:
                st.info(msg)
        return top_features

    with tabs[1]:
        try:
            with st.spinner("Generating summary plot..."):
                fig = summary_figure(dataset_bundle, max_display=top_k)
            st.image(_fig_to_image(fig), use_container_width=True)
            st.caption(
                "전체 샘플의 SHAP 분포. 색상은 피처 값의 크기."
                if language == "ko"
                else "SHAP distribution across samples. Color encodes feature value."
            )
        except Exception as e:
            st.warning(f"Summary plot failed: {e}")

    with tabs[2]:
        try:
            fig = feature_importance_figure(dataset_bundle, max_display=top_k)
            st.image(_fig_to_image(fig), use_container_width=True)
            st.caption(
                "평균 |SHAP 값| 기준 피처 중요도 (reject 클래스 기준)."
                if language == "ko"
                else "Mean |SHAP| importance for the reject class."
            )
        except Exception as e:
            st.warning(f"Feature importance failed: {e}")

    with tabs[3]:
        try:
            choices = ranked_feature_names(dataset_bundle, limit=20)
            if not choices:
                st.info("No features available.")
            else:
                label = "피처 선택" if language == "ko" else "Feature"
                feature = st.selectbox(label, options=choices, key="shap_dep_feature")
                with st.spinner("Generating dependency plot..."):
                    fig = dependency_figure(dataset_bundle, feature)
                st.image(_fig_to_image(fig), use_container_width=True)
                st.caption(
                    "선택한 피처 값에 따른 SHAP 값 분포."
                    if language == "ko"
                    else "SHAP value vs. selected feature value."
                )
        except Exception as e:
            st.warning(f"Dependency plot failed: {e}")

    return top_features


def render_ai_explanation(
    pred_class: int, top_features: List[Tuple[str, float]], language: Language
) -> None:
    header = "### AI 설명" if language == "ko" else "### AI Explanation"
    st.markdown("---")
    st.markdown(header)
    with st.spinner("Generating AI explanation..."):
        text = generate_underwriting_explanation(
            decision=decision_from_class(pred_class),
            pred_class=pred_class,
            top_features=top_features,
            language=language,
        )
    st.markdown(
        f"""
        <div style="padding:1rem;border-radius:0.5rem;background:#f8f9fa;
                    border-left:4px solid #4e79a7;white-space:pre-wrap;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )
