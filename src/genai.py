from __future__ import annotations
import os
from typing import Any, Literal

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

if load_dotenv is not None:
    # Load environment variables from a .env file if present.
    # This allows OPENAI_API_KEY (and others) to be configured without code changes.
    load_dotenv()

DecisionStyle = Literal["approve", "reject", "neutral"]


def _format_top_features(top_features: list[tuple[str, float]], k: int = 5) -> str:
    items = []
    for name, val in top_features[:k]:
        direction = "+" if val >= 0 else "-"
        items.append(f"- {name}: 영향 {direction}, 기여도={abs(val):.3f}")
    return "\n".join(items)


def generate_underwriting_explanation(
    decision: DecisionStyle,
    pred_class: int,
    top_features: list[tuple[str, float]],
    probs: Any | None = None,
    sample: dict[str, Any] | None = None,
    language: Literal["ko", "en"] = "ko",
) -> str:
    """Generate an underwriting-style explanation using an LLM.

    Falls back to a deterministic template if OPENAI_API_KEY or openai package is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if OpenAI is None or not api_key:
        # Fallback deterministic text
        header = {
            "approve": "[오프라인 모드] 인수 승인 설명(샘플)",
            "reject": "[오프라인 모드] 인수 거절 사유(샘플)",
            "neutral": "[오프라인 모드] 예측 결과 설명(샘플)",
        }[decision]
        body = _format_top_features(top_features)
        return f"{header}\n예측 클래스: {pred_class}\n주요 기여 특성:\n{body}\n(OPENAI_API_KEY가 설정되지 않아 샘플 설명을 제공합니다.)"

    client = OpenAI(api_key=api_key)

    role_instruction_ko = (
        "당신은 보험 언더라이터입니다. 콜센터 상담원에게 전달할 간결하고 전문적인 설명을 작성하세요. "
        "모델 예측 결과(클래스)와 주요 기여 특성을 바탕으로, "
        "결정 유형에 맞게 승인 사유 또는 거절 사유를 4~6문장으로 설명하세요. "
        "과도한 추정은 피하고, 데이터 기반 근거를 중심으로 작성하세요."
    )
    style_ko = {
        "approve": "인수승인 관점에서, 고객의 위험요인이 통제 가능하거나 낮은 수준임을 근거로 승인 사유를 제시하세요.",
        "reject": "인수거절 관점에서, 고객의 위험요인이 높거나 불확실성이 큰 부분을 중심으로 거절 사유를 제시하세요.",
        "neutral": "중립적 관점에서, 위험요인과 완화 요인을 균형 있게 설명하세요.",
    }[decision]

    top_feats_bullets = _format_top_features(top_features)
    probs_text = f"클래스 확률: {probs}" if probs is not None else ""

    sys_msg = {"role": "system", "content": role_instruction_ko}
    user_msg = {
        "role": "user",
        "content": (
            f"예측 클래스: {pred_class}\n"
            f"{probs_text}\n"
            f"주요 기여 특성:\n{top_feats_bullets}\n"
            f"요구 스타일: {style_ko}"
        ),
    }

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[sys_msg, user_msg],
            temperature=0.4,
            max_tokens=280,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:  # graceful fallback
        header = {
            "approve": "인수 승인 설명(임시)",
            "reject": "인수 거절 사유(임시)",
            "neutral": "예측 결과 설명(임시)",
        }[decision]
        body = _format_top_features(top_features)
        return f"{header}\n예측 클래스: {pred_class}\n주요 기여 특성:\n{body}\n(생성형 API 호출 실패: {e})"
