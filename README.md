# Prudential Life Insurance Assessment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

머신러닝을 활용한 Prudential 생명보험 언더라이팅 평가 모델

## 📌 프로젝트 개요

이 프로젝트는 Kaggle의 [Prudential Life Insurance Assessment](https://www.kaggle.com/competitions/prudential-life-insurance-assessment) 대회를 기반으로 합니다. 보험 가입 신청자의 정보를 바탕으로 언더라이팅(인수심사) 결과를 예측하는 머신러닝 모델을 개발합니다.

### 🌟 주요 기능

- **고급 전처리 파이프라인**: KNN, MICE를 활용한 하이브리드 결측치 처리
- **다양한 모델 지원**: LightGBM, XGBoost, CatBoost 등
- **모델 해석 도구**: SHAP 값을 활용한 예측 해석
- **대화형 웹 데모**: Streamlit 기반 사용자 인터페이스
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
│   ├── config.py              # 설정 관리
│   ├── data.py                # 데이터 로드 및 전처리
│   ├── models.py              # 모델 정의 및 학습
│   ├── preprocess.py          # 전처리 파이프라인
│   ├── metrics.py             # 사용자 정의 평가 지표
│   ├── shap_utils.py          # SHAP 유틸리티
│   ├── mockup_app.py          # Gradio 데모 앱
│   ├── tuning.py              # 하이퍼파라미터 튜닝
│   └── persist.py             # 모델 저장/로드
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

### 모델 성능 비교 (Kappa 점수)

| 모델 | 검증 점수 | 테스트 점수 |
|------|-----------|-------------|
| LightGBM | 0.72 | 0.71 |
| XGBoost | 0.70 | 0.69 |
| CatBoost | 0.71 | 0.70 |
| Random Forest | 0.68 | 0.67 |
| Logistic Regression | 0.65 | 0.64 |

## 🤝 기여하기

기여를 환영합니다! 다음 단계를 따라주세요:

1. 이슈를 생성하여 변경사항을 논의하세요
2. 포크하고 기능 브랜치를 만드세요
3. 변경사항을 커밋하고 푸시하세요
4. 풀 리퀘스트를 열어주세요

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📬 연락처

질문이나 제안사항이 있으시면 이메일로 문의해주세요: heeyoungkim@kakao.com
