import re
from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = PROJECT_ROOT / "prudential-life-insurance-assessment-python.ipynb"
OUT_NB_PATH = PROJECT_ROOT / "prudential-life-insurance-assessment-python-enhanced.ipynb"
COMPANION_MD = PROJECT_ROOT / "notebook_companion.md"


def split_markdown_sections(md_text: str):
    # Split by lines consisting of ---
    parts = re.split(r"^---\s*$", md_text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def build_markdown_cells(md_text: str):
    sections = split_markdown_sections(md_text)
    cells = []
    for sec in sections:
        cells.append(nbf.v4.new_markdown_cell(sec))
    return cells


def fix_data_path_in_code(source: str) -> str:
    # Replace extract_path assignment to use 'data/' (relative)
    # Handles both single and double quotes and optional spaces
    pattern = r"^(\s*extract_path\s*=\s*)([\"\']).*?\2\s*$"
    repl = r"\1'data/'"
    lines = source.splitlines()
    new_lines = []
    for line in lines:
        if re.match(pattern, line):
            new_lines.append(re.sub(pattern, repl, line))
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def build_code_flow_cells():
    """Return a list of markdown + code cells covering the full workflow."""
    cells = []

    # 0) Setup & Imports
    cells.append(nbf.v4.new_markdown_cell("### Code: 환경 설정 및 임포트"))
    code_imports = (
        """
# 경로 추가(필요시) 및 재현성 시딩
import os, sys
sys.path.append(os.getcwd())

from src.config import seed_everything
from src.data import load_csvs, split_X_y
from src.models import make_pipelines
from src.metrics import evaluate_multiclass
from src.preprocess import build_preprocessor_from_df
from src.tuning import tune_lightgbm
from src.shap_utils import top_contributors_for_instance
from src.mockup_app import launch_demo_with_dataset

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
import pandas as pd

seed_everything(42)
        """
    ).strip()
    cells.append(nbf.v4.new_code_cell(code_imports))

    # 1) Load Data
    cells.append(nbf.v4.new_markdown_cell("### Code: 데이터 로드 및 분리"))
    code_data = (
        """
train_df, test_df, sample_df = load_csvs()  # data/ 경로 사용
X, y = split_X_y(train_df)
X.shape, y.shape
        """
    ).strip()
    cells.append(nbf.v4.new_code_cell(code_data))

    # 2) Baseline CV
    cells.append(nbf.v4.new_markdown_cell("### Code: 베이스라인 모델 비교 (5-Fold, QWK 포함)"))
    code_cv = (
        """
pipelines = make_pipelines(train_df)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, pipe in pipelines.items():
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
    results[name] = evaluate_multiclass(y, y_pred)

pd.DataFrame([{**{"model": k}, **{m: v for m, v in vals.items() if m in ("accuracy", "macro_f1", "qwk")}} for k, vals in results.items()])
        """
    ).strip()
    cells.append(nbf.v4.new_code_cell(code_cv))

    # 3) Optuna Tuning for LightGBM
    cells.append(nbf.v4.new_markdown_cell("### Code: LightGBM 하이퍼파라미터 튜닝 (Optuna)"))
    code_tune = (
        """
best_model, best_params = tune_lightgbm(train_df, X, y, n_trials=30, n_splits=5)
best_params
        """
    ).strip()
    cells.append(nbf.v4.new_code_cell(code_tune))

    # 4) Fit Final Pipeline & SHAP Example
    cells.append(nbf.v4.new_markdown_cell("### Code: 최종 파이프라인 적합 및 SHAP 기여도 예시"))
    code_shap = (
        """
pre_tree = build_preprocessor_from_df(train_df, for_linear=False)
final_pipe = Pipeline([("pre", pre_tree), ("clf", best_model)])
final_pipe.fit(X, y)

info = top_contributors_for_instance(final_pipe, X, index=0, top_k=8)
info
        """
    ).strip()
    cells.append(nbf.v4.new_code_cell(code_shap))

    # 5) Gradio Demo
    cells.append(nbf.v4.new_markdown_cell("### Code: Gradio Mock-up 데모 (원할 때 실행)"))
    code_gradio = (
        """
# 주의: 실행 시 로컬 서버가 열리므로, 데모 확인 후 셀 정지/재시작이 필요할 수 있습니다.
# launch_demo_with_dataset(final_pipe, X)
        """
    ).strip()
    cells.append(nbf.v4.new_code_cell(code_gradio))

    return cells


def process_notebook(nb):
    # Prepend markdown cells
    md_text = COMPANION_MD.read_text(encoding="utf-8")
    md_cells = build_markdown_cells(md_text)

    # Fix path in code cells
    for cell in nb.cells:
        if cell.get("cell_type") == "code" and isinstance(cell.get("source"), str):
            cell["source"] = fix_data_path_in_code(cell["source"])

    # Build and append sectioned code flow
    code_cells = build_code_flow_cells()

    nb.cells = md_cells + code_cells + nb.cells
    return nb


def main():
    assert NB_PATH.exists(), f"Notebook not found: {NB_PATH}"
    assert COMPANION_MD.exists(), f"Companion markdown not found: {COMPANION_MD}"

    nb = nbf.read(NB_PATH, as_version=4)
    nb = process_notebook(nb)
    nbf.write(nb, OUT_NB_PATH)
    print(f"Enhanced notebook written to: {OUT_NB_PATH}")


if __name__ == "__main__":
    main()
