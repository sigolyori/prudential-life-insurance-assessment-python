# Prudential Life Insurance Assessment - Deployment Guide

이 문서는 Prudential Life Insurance Assessment 모델을 로컬에서 실행하거나 Hugging Face Spaces에 배포하는 방법을 설명합니다.

## 로컬에서 실행하기

### 사전 요구사항
- Python 3.10 이상
- pip (Python 패키지 관리자)

### 1. 저장소 클론
```bash
git clone https://github.com/yourusername/prudential-life-insurance-assessment-python.git
cd prudential-life-insurance-assessment-python
```

### 2. 가상 환경 설정 및 활성화
```bash
# 가상 환경 생성
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
# source .venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements-deploy.txt
```

### 4. 애플리케이션 실행
```bash
python app.py
```

웹 브라우저에서 `http://localhost:7860`으로 접속하여 애플리케이션을 사용할 수 있습니다.

## Docker를 사용한 실행

### 1. Docker 이미지 빌드
```bash
docker build -t prudential-insurance-app .
```

### 2. Docker 컨테이너 실행
```bash
docker run -p 7860:7860 prudential-insurance-app
```

## Hugging Face Spaces에 배포하기

1. Hugging Face 계정에 로그인하고 새로운 Space를 생성합니다.
2. Space 설정에서 "Docker"를 선택합니다.
3. 이 저장소를 연결하거나 파일을 업로드합니다.
4. `app.py`가 메인 애플리케이션 파일로 인식됩니다.
5. 필요한 환경 변수가 있는 경우 Space 설정에서 추가합니다.
6. Space를 생성하고 배포를 기다립니다.

## 환경 변수

필요한 경우 `.env` 파일을 생성하여 다음 환경 변수를 설정할 수 있습니다:

```
# 모델 및 데이터 경로
MODEL_PATH=data/processed/final_pipe.joblib
DATA_PATH=data/raw/train.csv

# 애플리케이션 설정
PORT=7860
HOST=0.0.0.0
```

## 문제 해결

- **모델 로드 오류**: `final_pipe.joblib` 파일이 올바른 경로에 있는지 확인하세요.
- **의존성 문제**: `requirements-deploy.txt`에 모든 필요한 패키지가 포함되어 있는지 확인하세요.
- **포트 충돌**: 다른 애플리케이션이 7860 포트를 사용 중인 경우, `app.py`에서 포트 번호를 변경하세요.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
