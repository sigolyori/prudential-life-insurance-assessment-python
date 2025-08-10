from __future__ import annotations
import gradio as gr
import pandas as pd
from typing import Optional

import io
import numpy as np
from PIL import Image

from .shap_utils import (
    top_contributors_for_instance,
    prime_shap_cache,
    waterfall_figure_for_instance,
)
from .genai import generate_underwriting_explanation


def _render_explanation_ko(sample_idx: int, pred_class: int, top_feats: list[tuple[str, float]]) -> str:
    parts = [
        f"사례 #{sample_idx}에 대한 예측 결과는 클래스 {pred_class} 입니다.",
        "모델 결정에 크게 기여한 특성은 다음과 같습니다:",
    ]
    for name, val in top_feats[:5]:
        direction = "+" if val >= 0 else "-"
        parts.append(f"- {name}: 영향 방향 {direction} (기여도={abs(val):.3f})")
    parts.append("해당 결과는 데이터 기반 자동화 판단으로, 실제 언더라이팅 시에는 추가 검토가 필요할 수 있습니다.")
    return "\n".join(parts)


def launch_demo_with_dataset(pipeline, X: pd.DataFrame, class_names: Optional[list[str]] = None):
    n = len(X)

    # Pre-warm and cache the SHAP explainer (LightGBM-only). This also validates the model.
    # If the model is not LightGBM, this will raise a ValueError early.
    prime_shap_cache(pipeline, X)

    def _fig_to_image(fig) -> Image.Image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def predict_and_explain(idx: int):
        try:
            idx = int(max(0, min(idx, n - 1)))
            info = top_contributors_for_instance(pipeline, X, idx)
            pred_class = info["pred_class"]
            # Build SHAP waterfall figure and convert to numpy image
            fig = waterfall_figure_for_instance(pipeline, X, idx, top_k=8)
            img = _fig_to_image(fig)

            # Auto-map decision style from predicted class: 1 -> reject, others -> approve
            decision = "reject" if int(pred_class) == 1 else "approve"
            gen_text = generate_underwriting_explanation(
                decision=decision,
                pred_class=pred_class,
                top_features=info["top_features"],
                probs=info.get("probs"),
            )
            return str(pred_class), img, gen_text
        except Exception as e:
            return "", None, f"오류가 발생했습니다: {e}"

    with gr.Blocks() as demo:
        gr.Markdown("## 언더라이팅 예측 Mock-up")
        gr.Markdown("샘플 인덱스를 선택한 후 예측 및 설명을 생성합니다. (클래스 1은 인수거절, 나머지는 인수승인 관점)")

        # Top tier: index selector
        with gr.Row():
            idx = gr.Slider(0, n - 1, value=0, step=1, label="샘플 인덱스")
            btn = gr.Button("예측 및 설명 생성")

        # Middle tier: predicted class
        pred = gr.Textbox(label="예측 클래스", interactive=False)

        # Bottom tier: left (waterfall), right (LLM explanation)
        with gr.Row():
            chart = gr.Image(label="SHAP Waterfall", type="pil")
            expl = gr.Textbox(label="생성형 설명", lines=12)

        btn.click(fn=predict_and_explain, inputs=[idx], outputs=[pred, chart, expl])
    demo.launch()
