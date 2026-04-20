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
    class_label = "예측 클래스" if language == "ko" else "Predicted class"
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:1rem;margin:0.25rem 0 1rem 0;">
          <span style="background:{color};color:white;padding:0.35rem 0.9rem;
                       border-radius:999px;font-weight:600;font-size:1rem;">
            {label}
          </span>
          <span style="color:#555;font-size:0.95rem;">
            {class_label}: <b>Class {pred_class}</b>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_chart(pred_proba: np.ndarray, language: Language) -> None:
    header = "#### 클래스 확률" if language == "ko" else "#### Class probabilities"
    st.markdown(header)
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


def render_shap_panel(
    pipeline, sample: pd.DataFrame, top_k: int, language: Language
) -> List[Tuple[str, float]]:
    header = "#### 피처 영향 분석" if language == "ko" else "#### Feature Impact"
    st.markdown(header)

    clf = _extract_estimator(pipeline)
    if not _is_lgbm(clf):
        msg = (
            "SHAP Waterfall은 LightGBM 모델에서만 지원됩니다."
            if language == "ko"
            else "SHAP Waterfall is only available for LightGBM models."
        )
        st.info(msg)
        return []

    try:
        with st.spinner("Computing SHAP contributions..."):
            contrib_info = top_contributors_for_instance(pipeline, sample, index=0, top_k=top_k)
            top_features = contrib_info.get("top_features", [])
            fig = waterfall_figure_for_instance(pipeline, sample, index=0, top_k=top_k)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        st.image(Image.open(buf), use_container_width=True)

        caption = (
            "파란 막대는 예측을 높이는 방향, 빨간 막대는 낮추는 방향의 피처 기여도입니다."
            if language == "ko"
            else "Blue bars push the prediction up; red bars push it down."
        )
        st.caption(caption)
        return top_features
    except Exception as e:
        st.warning(f"Could not generate SHAP chart: {e}")
        return []


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
