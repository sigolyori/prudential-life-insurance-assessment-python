# Prudential Life Insurance Assessment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

머신러닝을 활용한 Prudential 생명보험 언더라이팅 평가 모델

## 📌 프로젝트 개요

이 프로젝트는 Kaggle의 [Prudential Life Insurance Assessment](https://www.kaggle.com/competitions/prudential-life-insurance-assessment) 대회를 기반으로 합니다. 보험 가입 신청자의 정보를 바탕으로 언더라이팅(인수심사) 결과를 예측하는 머신러닝 모델을 개발합니다.

### 🌟 주요 기능

- **고급 전처리 파이프라인**: KNN, MICE를 활용한 하이브리드 결측치 처리
- **다양한 모델 지원**: LightGBM, Random Forest, Logistic Regression, SVM (기본값: LightGBM)
- **빠른 모델 해석**: LightGBM 네이티브 `pred_contrib` 기반 SHAP + Waterfall 차트
- **GenAI 설명 생성**: OpenAI API를 이용한 한국어/영어 언더라이팅 사유 자동 생성 (오프라인 폴백 지원)
- **대화형 웹 데모**: Streamlit 대시보드 (Plotly 확률 차트, 다국어 토글, Top-K 슬라이더)
- **자동화된 CI/CD**: 테스트, 린트, 배포 자동화

## 🚀 시작하기

### 사전 요구사항

- Python 3.10+
- pip (Python 패키지 관리자)
- Git

### 설치 방법

1. **저장소 클론**
   ```bash
   git clone https://github.com/sigolyori/prudential-life-insurance-assessment-python.git
   cd prudential-life-insurance-assessment-python
   ```

2. **가상 환경 설정**
   ```bash
   # 가상 환경 생성
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   # source .venv/bin/activate
   ```

3. **의존성 설치**
   ```bash
   # 개발 환경
   pip install -e ".[dev]"
   
   # 또는 프로덕션 환경
   # pip install -e .
   ```

## 📊 데이터 준비

1. [Kaggle](https://www.kaggle.com/competitions/prudential-life-insurance-assessment/data)에서 데이터 다운로드
2. `data/raw/` 디렉토리에 다음 파일들을 복사:
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`

## 🧪 사용 방법

### 모델 학습 및 평가

```python
from src.models import train_model, evaluate_model
from src.data import load_data

# 데이터 로드
X_train, y_train, X_test = load_data()

# 모델 학습
model = train_model(X_train, y_train)

# 모델 평가
results = evaluate_model(model, X_test, y_test)
print(results)
```

### 웹 데모 실행

#### Gradio 앱으로 실행
```bash
# 로컬에서 실행
python app.py

# 또는 개발 모드로 실행
python -m src.mockup_app
```

#### Streamlit 앱으로 실행 (권장)
```bash
# 의존성 설치
pip install -r requirements-streamlit.txt

# 로컬에서 실행
streamlit run streamlit_app.py
```

## 🛠 프로젝트 구조

```
prudential-life-insurance-assessment-python/
├── .github/                     # GitHub Actions 워크플로우
│   └── workflows/
│       ├── ci.yml              # CI 파이프라인
│       ├── cd.yml              # CD 파이프라인
│       └── deploy-blog.yml     # 블로그 배포
│
├── blog/                       # Quarto 블로그
│   ├── index.qmd              # 메인 블로그 포스트
│   ├── images/                # 블로그 이미지
│   └── data/                  # 블로그용 데이터
│
├── data/                       # 데이터 파일
│   ├── raw/                   # 원본 데이터 (Git LFS)
│   ├── processed/             # 전처리된 데이터
│   └── external/              # 외부 데이터
│
├── docs/                       # 문서화
│   ├── api/                   # API 문서
│   └── notebooks/             # 렌더링된 노트북
│
├── notebooks/                  # Jupyter 노트북
│   ├── exploratory/           # 탐색적 분석
│   └── modeling/              # 모델링 실험
│
├── reports/                    # 분석 결과
│   └── figures/               # 시각화 자료
│
├── src/                        # 소스 코드
│   ├── __init__.py
│   ├── config.py              # 설정/상수 관리 (경로, 클래스 수, 결정 규칙 등)
│   ├── data.py                # 데이터 로드
│   ├── preprocess.py          # 전처리 파이프라인 (KNN/MICE 하이브리드)
│   ├── models.py              # 모델 파이프라인 팩토리
│   ├── tuning.py              # Optuna 기반 LightGBM 튜닝
│   ├── metrics.py             # QWK(Quadratic Weighted Kappa) scorer
│   ├── shap_utils.py          # LightGBM pred_contrib 기반 SHAP 유틸리티
│   ├── persist.py             # 모델 저장/로드 + SHAP 캐시
│   ├── genai.py               # OpenAI 기반 언더라이팅 설명 생성
│   ├── mockup_app.py          # Gradio 데모 앱
│   └── ui/                    # Streamlit UI 컴포넌트
│       ├── __init__.py
│       ├── state.py           # 캐시된 리소스 로더
│       └── components.py      # 대시보드 재사용 컴포넌트
│
├── tests/                      # 단위 테스트
│   ├── test_data.py
│   ├── test_models.py
│   └── test_preprocess.py
│
├── .dockerignore              # Docker 빌드 컨텍스트 제외 파일
├── .gitignore                 # Git 추적 제외 파일
├── app.py                     # 배포용 Gradio 앱
├── streamlit_app.py           # Streamlit 웹 앱
├── Dockerfile                 # Docker 컨테이너 설정
├── pyproject.toml             # 프로젝트 메타데이터 및 의존성
├── README.md                  # 이 파일
└── requirements-deploy.txt    # 배포용 의존성
```

## 🚀 배포

### Streamlit Cloud에 배포 (추천)

1. [Streamlit Cloud](https://share.streamlit.io/)에 로그인
2. "New app" 클릭
3. GitHub 저장소 선택 후 다음 설정:
   - Repository: `yourusername/prudential-life-insurance-assessment-python`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. "Deploy!" 클릭

**👉 [실제 배포된 앱 바로가기](https://sigolyori-prudential-life-insurance-assess-streamlit-app-ddmkre.streamlit.app/)**

### Hugging Face Spaces에 배포

1. Hugging Face 계정 생성 및 로그인
2. 새로운 Space 생성 (Gradle SDK 선택)
3. 저장소 연결 또는 파일 업로드
4. 자동 배포 대기

### 로컬에서 Docker로 실행

```bash
# Docker 이미지 빌드
docker build -t prudential-insurance-app .

# 컨테이너 실행
docker run -p 7860:7860 prudential-insurance-app
```

## 📈 성능

### 모델 성능 비교 (Quadratic Weighted Kappa)

| 모델 | 검증 점수 |
|------|-----------|
| LightGBM (튜닝 후) | 0.72 |
| Random Forest | 0.68 |
| SVM (RBF) | 0.66 |
| Logistic Regression | 0.65 |

> Streamlit 대시보드는 `data/processed/final_pipe.joblib` 에 저장된 학습 완료 LightGBM 파이프라인을 사용합니다.

## 📬 연락처

질문이나 제안사항이 있으시면 이메일로 문의해주세요: heeyoungkim@kakao.com
