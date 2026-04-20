from __future__ import annotations

import os
from typing import Any, Iterable, Literal

from .config import DEFAULT_LANGUAGE, Language

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

DecisionStyle = Literal["approve", "reject", "neutral"]

_HEADERS = {
    "ko": {
        "approve": "[오프라인 모드] 인수 승인 설명(샘플)",
        "reject": "[오프라인 모드] 인수 거절 사유(샘플)",
        "neutral": "[오프라인 모드] 예측 결과 설명(샘플)",
    },
    "en": {
        "approve": "[Offline mode] Underwriting approval rationale (sample)",
        "reject": "[Offline mode] Underwriting rejection rationale (sample)",
        "neutral": "[Offline mode] Prediction rationale (sample)",
    },
}

_ROLE = {
    "ko": (
        "당신은 보험 언더라이터입니다. 콜센터 상담원에게 전달할 간결하고 전문적인 설명을 작성하세요. "
        "모델 예측 결과(클래스)와 주요 기여 특성을 바탕으로, 결정 유형에 맞게 승인 사유 또는 거절 사유를 "
        "4~6문장으로 설명하세요. 과도한 추정은 피하고, 데이터 기반 근거를 중심으로 작성하세요."
    ),
    "en": (
        "You are an insurance underwriter. Write a concise, professional explanation for a call-center agent. "
        "Based on the predicted class and the top contributing features, produce an approval or rejection "
        "rationale in 4-6 sentences. Avoid speculation; stay grounded in the provided evidence."
    ),
}

_STYLES = {
    "ko": {
        "approve": "인수승인 관점에서, 고객의 위험요인이 통제 가능하거나 낮은 수준임을 근거로 승인 사유를 제시하세요.",
        "reject": "인수거절 관점에서, 고객의 위험요인이 높거나 불확실성이 큰 부분을 중심으로 거절 사유를 제시하세요.",
        "neutral": "중립적 관점에서, 위험요인과 완화 요인을 균형 있게 설명하세요.",
    },
    "en": {
        "approve": "From an approval perspective, justify acceptance by noting manageable or low-severity risks.",
        "reject": "From a rejection perspective, focus on high or uncertain risk factors driving the decision.",
        "neutral": "From a neutral perspective, balance risk factors and mitigating factors.",
    },
}


def _format_top_features(top_features: Iterable[tuple[str, float]], k: int = 5) -> str:
    lines = []
    for name, val in list(top_features)[:k]:
        direction = "+" if val >= 0 else "-"
        lines.append(f"- {name}: {direction}, contribution={abs(val):.3f}")
    return "\n".join(lines)


def _fallback(
    decision: DecisionStyle,
    pred_class: int,
    top_features: list[tuple[str, float]],
    language: Language,
    note: str,
) -> str:
    header = _HEADERS[language][decision]
    body = _format_top_features(top_features)
    label = "예측 클래스" if language == "ko" else "Predicted class"
    feat_label = "주요 기여 특성" if language == "ko" else "Top contributing features"
    return f"{header}\n{label}: {pred_class}\n{feat_label}:\n{body}\n({note})"


def generate_underwriting_explanation(
    decision: DecisionStyle,
    pred_class: int,
    top_features: list[tuple[str, float]],
    probs: Any | None = None,
    sample: dict[str, Any] | None = None,
    language: Language = DEFAULT_LANGUAGE,
) -> str:
    """Generate an underwriting explanation via OpenAI, with a deterministic fallback."""
    api_key = os.getenv("OPENAI_API_KEY")
    if OpenAI is None or not api_key:
        note = (
            "OPENAI_API_KEY가 설정되지 않아 샘플 설명을 제공합니다."
            if language == "ko"
            else "Set OPENAI_API_KEY to enable live explanations."
        )
        return _fallback(decision, pred_class, top_features, language, note)

    client = OpenAI(api_key=api_key)
    feats = _format_top_features(top_features)
    probs_text = (
        (f"클래스 확률: {probs}" if language == "ko" else f"Class probabilities: {probs}")
        if probs is not None
        else ""
    )
    pred_label = "예측 클래스" if language == "ko" else "Predicted class"
    feats_label = "주요 기여 특성" if language == "ko" else "Top contributing features"
    style_label = "요구 스타일" if language == "ko" else "Required style"

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": _ROLE[language]},
                {
                    "role": "user",
                    "content": (
                        f"{pred_label}: {pred_class}\n"
                        f"{probs_text}\n"
                        f"{feats_label}:\n{feats}\n"
                        f"{style_label}: {_STYLES[language][decision]}"
                    ),
                },
            ],
            temperature=0.4,
            max_tokens=320,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        note = f"생성형 API 호출 실패: {e}" if language == "ko" else f"GenAI call failed: {e}"
        return _fallback(decision, pred_class, top_features, language, note)
