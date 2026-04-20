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
        "당신은 콜센터 상담원을 지원하는 실시간 AI 어시스턴트입니다. "
        "상담원은 지금 고객과 전화 통화 중이며, 당신의 메시지를 보는 즉시 "
        "고객에게 전달해야 합니다. "
        "모델 예측 결과와 주요 기여 변수를 바탕으로 아래 세 가지를 순서대로 작성하세요.\n"
        "① 고객 전달 멘트: 상담원이 고객에게 바로 읽어줄 수 있는 1~2문장. "
        "고객에게 직접 말하는 말투(예: '고객님, ~')로 작성하세요.\n"
        "② 핵심 근거: 고객이 '왜요?'라고 물을 때 상담원이 답할 수 있는 "
        "1~2가지 이유. 전문 용어 없이 쉬운 표현으로 작성하세요.\n"
        "③ 다음 안내: 상담원이 통화를 마무리할 때 고객에게 전달할 "
        "후속 절차 또는 대안을 1문장으로 작성하세요.\n"
        "과도한 추정은 피하고, 데이터 기반 근거만 사용하세요."
    ),
    "en": (
        "You are a real-time AI assistant supporting a call-center agent. "
        "The agent is on the phone with a customer right now and will relay "
        "your message immediately. "
        "Based on the model prediction and top contributing features, "
        "write the following three parts in order:\n"
        "① Customer script: 1-2 sentences the agent can read aloud directly "
        "to the customer (use second-person: 'You have been...').\n"
        "② Key reason: 1-2 plain-language reasons the agent can use if the "
        "customer asks 'why'. Avoid technical jargon.\n"
        "③ Next step: One sentence guiding the agent on how to close the call "
        "or what to offer next.\n"
        "Avoid speculation; stay grounded in the provided evidence."
    ),
}

_STYLES = {
    "ko": {
        "approve": (
            "인수승인 결과를 고객에게 자연스럽고 긍정적으로 전달하세요. "
            "고객이 안심하고 기뻐할 수 있도록 따뜻한 말투를 사용하고, "
            "승인된 이유를 쉬운 말로 1가지만 언급하세요. "
            "다음 절차(서류 제출, 계약 진행 등)로 자연스럽게 연결하여 마무리하세요."
        ),
        "reject": (
            "인수거절 결과를 정중하고 명확하게 전달하되, "
            "고객이 불쾌하지 않도록 배려하는 말투를 사용하세요. "
            "거절 이유는 과도하게 상세히 말하지 말고 핵심 사유 1가지만 "
            "이해하기 쉬운 표현으로 언급하세요. "
            "재심사 가능성, 조건부 가입, 또는 대안 상품 안내 등 "
            "고객에게 도움이 될 다음 단계를 반드시 제시하세요."
        ),
        "neutral": (
            "심사를 위해 추가 정보가 필요한 상황입니다. "
            "고객에게 현재 결정이 보류 중임을 안내하고, "
            "상담원이 고객에게 요청해야 할 추가 서류나 정보가 무엇인지 "
            "구체적으로 명시하세요. "
            "고객이 불안해하지 않도록 처리 예상 일정도 함께 안내하세요."
        ),
    },
    "en": {
        "approve": (
            "Deliver the approval result warmly and positively. "
            "Use reassuring language so the customer feels confident. "
            "Mention only one simple reason for approval in plain terms. "
            "Close by guiding toward the next step (e.g., document submission, "
            "contract signing)."
        ),
        "reject": (
            "Deliver the rejection result politely and clearly. "
            "Be considerate in tone so the customer does not feel dismissed. "
            "State only one core reason in plain language without excessive detail. "
            "Always offer a next step: re-evaluation eligibility, conditional "
            "coverage, or an alternative product."
        ),
        "neutral": (
            "The decision is pending due to insufficient information. "
            "Inform the customer that processing is on hold, specify exactly "
            "what additional documents or information are needed, and provide "
            "an estimated timeline so the customer feels supported."
        ),
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
            max_tokens=350,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        note = f"생성형 API 호출 실패: {e}" if language == "ko" else f"GenAI call failed: {e}"
        return _fallback(decision, pred_class, top_features, language, note)
