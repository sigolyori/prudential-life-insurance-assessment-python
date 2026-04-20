"""Streamlit dashboard for the Prudential underwriting assessment pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent))

from src.ui import (
    get_model_info,
    load_resources,
    render_ai_explanation,
    render_decision_badge,
    render_probability_chart,
    render_shap_panel,
    render_sidebar,
)


st.set_page_config(
    page_title="Prudential Life Insurance Assessment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton>button { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _align_to_model(model, sample: pd.DataFrame) -> pd.DataFrame:
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return sample
    aligned = sample.copy()
    for col in expected:
        if col not in aligned.columns:
            aligned[col] = 0
    return aligned[list(expected)]


def _predict(model, sample: pd.DataFrame):
    aligned = _align_to_model(model, sample)
    pred_proba = model.predict_proba(aligned)[0]
    pred_class = int(np.argmax(pred_proba)) + 1
    return pred_class, pred_proba, aligned


def main() -> None:
    st.title("Prudential Life Insurance Underwriting Assessment")
    st.markdown(
        "LightGBM 모델, SHAP 기여도 분석, GenAI 설명을 결합한 End-to-End 인수심사 데모입니다. "
        "사이드바에서 테스트 케이스와 설정을 조정해보세요."
    )

    resources = load_resources()
    model_info = get_model_info(resources.pipeline)
    settings = render_sidebar(len(resources.X_test), model_info)

    sample = resources.X_test.iloc[[settings["sample_idx"]]]
    try:
        pred_class, pred_proba, _ = _predict(resources.pipeline, sample)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("Prediction")
    render_decision_badge(pred_class, settings["language"])

    col_prob, col_shap = st.columns(2)
    with col_prob:
        render_probability_chart(pred_proba, settings["language"])
    with col_shap:
        top_features = render_shap_panel(
            resources.pipeline, sample, settings["top_k"], settings["language"]
        )

    render_ai_explanation(pred_class, top_features, settings["language"])


if __name__ == "__main__":
    main()
