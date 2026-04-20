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
    is_binary = (
        class_labels is not None
        and len(class_labels) == 2
        and set(int(c) for c in class_labels) == {0, 1}
    )

    header = "#### 클래스 확률" if language == "ko" else "#### Class probabilities"
    st.markdown(header)

    if language == "ko":
        lead = (
            "모델이 **각 결과에 대해 얼마나 확신하는지**를 나타냅니다. "
            "막대가 가장 높은 클래스가 최종 예측입니다."
        )
    else:
        lead = (
            "How confident the model is in **each possible outcome**. "
            "The tallest bar is the final prediction."
        )
    st.caption(lead)

    if class_labels is not None and len(class_labels) == len(pred_proba):
        if is_binary:
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
        height=440,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Probability", range=[0, min(1.0, float(pred_proba.max()) * 1.25)], tickformat=".0%"),
        xaxis=dict(title=""),
    )
    st.plotly_chart(fig, use_container_width=True)

    if language == "ko":
        expander_label = "📖 이 클래스들은 무엇인가요?"
        if is_binary:
            body = (
                "- **승인(0)**: 보험 인수 가능. 대부분의 정상 신청이 여기에 해당합니다.\n"
                "- **거절(1)**: 보험 인수가 어렵다고 판단된 신청.\n\n"
                "모델은 두 결과의 확률을 계산하고 더 높은 쪽을 예측으로 선택합니다. "
                "확률 차이가 작으면 경계선 사례이므로 상담원의 추가 확인이 권장됩니다."
            )
        else:
            body = (
                "Prudential 공개 데이터의 `Response` 변수는 1~8의 8단계로 기록됩니다. "
                "정확한 정의는 공개되어 있지 않지만, 본 프로젝트에서는 "
                "**Class 1 = 인수거절**, **Class 2~8 = 인수승인**으로 가정합니다 "
                "(`src/data.py`의 `binarize_target` 참고).\n\n"
                "각 막대는 모델이 해당 클래스라고 판단한 확률이며, 합은 100%입니다. "
                "주황색 막대는 최종 예측 클래스입니다."
            )
    else:
        expander_label = "📖 What do these classes mean?"
        if is_binary:
            body = (
                "- **Approve (0)**: Eligible for underwriting. Most typical applications.\n"
                "- **Reject (1)**: Flagged as not underwritable.\n\n"
                "The model outputs a probability for each outcome and picks the larger one. "
                "If the gap is small, the case is borderline and warrants human review."
            )
        else:
            body = (
                "Prudential's public `Response` variable has 8 ordinal levels. "
                "Their exact meanings are not published, so this project assumes "
                "**Class 1 = reject** and **Class 2-8 = approve** "
                "(see `binarize_target` in `src/data.py`).\n\n"
                "Each bar is the model's probability for that class; probabilities "
                "sum to 100%. The orange bar is the predicted class."
            )
    with st.expander(expander_label):
        st.markdown(body)


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
        if language == "ko":
            with st.expander("📖 이 차트는 무엇을 보여주나요?", expanded=False):
                st.markdown(
                    "**지금 보고 있는 이 한 명의 고객**에 대해 모델이 어떻게 결정을 내렸는지 "
                    "단계별로 보여주는 차트입니다.\n\n"
                    "- 차트 아래쪽의 **기준값(base value)** 은 모든 고객의 평균 예측입니다.\n"
                    "- 각 피처(나이, BMI, 병력 등)가 **기준값에서 얼마나 예측을 밀어올렸거나 내렸는지**를 막대로 표시합니다.\n"
                    "- **빨강**: 거절 확률을 높이는 쪽으로 기여. **파랑**: 승인 쪽으로 기여.\n"
                    "- 막대 길이가 길수록 이 고객의 결정에 더 큰 영향을 준 피처입니다.\n\n"
                    "👉 **이 한 건의 결정을 고객에게 설명할 때 쓰기 가장 좋은 차트입니다.**"
                )
        else:
            with st.expander("📖 What does this chart show?", expanded=False):
                st.markdown(
                    "A step-by-step view of how the model decided **for this one customer**.\n\n"
                    "- The **base value** at the bottom is the average prediction across all customers.\n"
                    "- Each feature (age, BMI, medical history, ...) is drawn as a bar showing "
                    "**how much it pushed the prediction up or down from that base**.\n"
                    "- **Red**: pushes toward reject. **Blue**: pushes toward approve.\n"
                    "- Longer bars = bigger influence on this specific decision.\n\n"
                    "👉 **This is the best chart to use when explaining a single case to a customer.**"
                )
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
        if language == "ko":
            with st.expander("📖 이 차트는 무엇을 보여주나요?", expanded=False):
                st.markdown(
                    "**전체 고객 데이터에서** 각 피처가 모델 예측에 어떤 영향을 주는지 "
                    "한눈에 보는 차트입니다.\n\n"
                    "- **점 하나 = 고객 한 명**. 여러 고객의 결과가 겹쳐서 보입니다.\n"
                    "- **가로축**: SHAP 값 (오른쪽 = 거절 방향, 왼쪽 = 승인 방향).\n"
                    "- **색**: 그 피처의 값 크기 (🔴 빨강 = 높은 값, 🔵 파랑 = 낮은 값).\n"
                    "- 위쪽에 있는 피처일수록 전반적으로 영향력이 큽니다.\n\n"
                    "👉 예: 빨간 점들이 오른쪽에 몰려 있다면 **\"그 피처 값이 높을수록 거절 위험이 커진다\"** 는 뜻입니다."
                )
        else:
            with st.expander("📖 What does this chart show?", expanded=False):
                st.markdown(
                    "A bird's-eye view of **how each feature affects predictions across all customers**.\n\n"
                    "- **Each dot = one customer.** Many dots pile up to show the distribution.\n"
                    "- **X-axis**: SHAP value (right = pushes toward reject, left = pushes toward approve).\n"
                    "- **Color**: the feature's own value (🔴 red = high, 🔵 blue = low).\n"
                    "- Features near the top have the largest overall influence.\n\n"
                    "👉 Example: if red dots cluster on the right, **\"higher values of that feature "
                    "raise the reject risk.\"**"
                )
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
        if language == "ko":
            with st.expander("📖 이 차트는 무엇을 보여주나요?", expanded=False):
                st.markdown(
                    "**어떤 피처가 모델 전체에서 가장 중요한지** 순서대로 보여주는 막대그래프입니다.\n\n"
                    "- 각 피처의 SHAP 값 **절댓값 평균**을 막대 길이로 그립니다 "
                    "(방향에 상관없이 \"얼마나 크게 영향을 미쳤나?\").\n"
                    "- 위에 있을수록 중요한 피처입니다.\n"
                    "- 현재는 **거절(reject) 클래스 기준** 중요도를 보여주어, "
                    "\"어떤 변수가 거절 위험 판단에 가장 큰 역할을 하는가?\"를 알 수 있습니다.\n\n"
                    "👉 Summary 차트와 달리 방향(위험↑/↓)은 알 수 없고, **크기만** 비교합니다."
                )
        else:
            with st.expander("📖 What does this chart show?", expanded=False):
                st.markdown(
                    "A ranked bar chart of **which features matter most overall**.\n\n"
                    "- Bar length = **mean absolute SHAP value** across all customers "
                    "(\"how big is this feature's influence, regardless of direction?\").\n"
                    "- Top features are the most important ones.\n"
                    "- Computed for the **reject class**, so you see \"which variables "
                    "drive reject-risk the most?\"\n\n"
                    "👉 Unlike the Summary chart, this one only shows **magnitude**, not direction."
                )
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
        if language == "ko":
            with st.expander("📖 이 차트는 무엇을 보여주나요?", expanded=False):
                st.markdown(
                    "**특정 피처 하나의 값이 바뀔 때 모델 예측이 어떻게 바뀌는지** "
                    "산점도로 보여줍니다.\n\n"
                    "- **가로축**: 선택한 피처의 실제 값.\n"
                    "- **세로축**: 그 값이 만든 SHAP 기여도 (양수 = 거절 방향, 음수 = 승인 방향).\n"
                    "- 점이 우상향하면 **\"값이 클수록 거절 위험↑\"**, 우하향하면 **\"값이 클수록 승인 경향↑\"** 입니다.\n"
                    "- 관계가 직선이 아니라 **비선형(꺾임, U자형 등)** 일 수도 있습니다.\n\n"
                    "👉 왼쪽 드롭다운에서 궁금한 피처를 골라 \"이 변수는 어떤 구간에서 위험해지나?\"를 확인하세요."
                )
        else:
            with st.expander("📖 What does this chart show?", expanded=False):
                st.markdown(
                    "A scatter plot of **how a single feature's value affects the prediction**.\n\n"
                    "- **X-axis**: the actual value of the chosen feature.\n"
                    "- **Y-axis**: the SHAP contribution it produced "
                    "(positive = reject direction, negative = approve direction).\n"
                    "- Rising points = **\"higher values → more reject-risk\"**. "
                    "Falling points = \"higher values → more approve-leaning\".\n"
                    "- The relationship may be **non-linear** (kinks, U-shapes, plateaus).\n\n"
                    "👉 Pick a feature from the dropdown to see **which value ranges "
                    "turn risky** for that variable."
                )
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


_MARKERS = ("①", "②", "③")


def _parse_three_parts(text: str) -> tuple[str, str, str] | None:
    """Split an explanation on ①②③ markers. Returns None if any marker is
    missing or out of order so the caller can fall back to the raw text."""
    if not all(m in text for m in _MARKERS):
        return None
    positions = [text.index(m) for m in _MARKERS]
    if not (positions[0] < positions[1] < positions[2]):
        return None

    segments: list[str] = []
    for i, marker in enumerate(_MARKERS):
        start = positions[i] + len(marker)
        end = positions[i + 1] if i + 1 < len(_MARKERS) else len(text)
        seg = text[start:end].strip()
        colon = seg.find(":")
        if 0 < colon < 40 and "\n" not in seg[:colon]:
            seg = seg[colon + 1 :].strip()
        segments.append(seg)
    return segments[0], segments[1], segments[2]


def render_ai_explanation(
    pred_class: int, top_features: List[Tuple[str, float]], language: Language
) -> None:
    st.markdown("---")
    st.markdown("### AI 설명" if language == "ko" else "### AI Explanation")

    with st.spinner("Generating AI explanation..."):
        text = generate_underwriting_explanation(
            decision=decision_from_class(pred_class),
            pred_class=pred_class,
            top_features=top_features,
            language=language,
        )

    parts = _parse_three_parts(text)
    if parts is None:
        st.markdown(
            f"""
            <div style="padding:1rem;border-radius:0.5rem;background:#f8f9fa;
                        border-left:4px solid #4e79a7;white-space:pre-wrap;">
                {text}
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    script, reason, next_step = parts
    decision = decision_from_class(pred_class)

    script_header = (
        "🎙️ 고객 전달 멘트" if language == "ko" else "🎙️ Customer script"
    )
    reason_header = "💡 핵심 근거" if language == "ko" else "💡 Key reason"
    next_header = "➡️ 다음 안내" if language == "ko" else "➡️ Next step"

    st.markdown(f"**{script_header}**")
    if decision == "approve":
        st.success(script)
    elif decision == "reject":
        st.error(script)
    else:
        st.info(script)

    st.markdown(f"**{reason_header}**")
    st.markdown(reason)

    st.markdown("---")
    st.markdown(f"**{next_header}**")
    st.markdown(next_step)
