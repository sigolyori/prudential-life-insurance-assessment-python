# 데이터 디렉토리 구조

이 디렉토리에는 Prudential Life Insurance Assessment 프로젝트와 관련된 모든 데이터 파일이 포함되어 있습니다.

## 디렉토리 구조

```
data/
├── raw/                  # 원본, 변경되지 않은 데이터
│   ├── train.csv         # 학습 데이터 (원본)
│   └── test.csv          # 테스트 데이터 (원본)
│   └── sample_submission.csv  # 제출 샘플 파일
│
├── processed/            # 전처리된 데이터 및 모델
│   └── final_pipe.joblib # 학습된 파이프라인 모델
│
└── external/             # 외부 데이터 소스 (사용되는 경우)
```

## 데이터 설명

### 원본 데이터

- **train.csv**: 모델 학습을 위한 레이블이 포함된 학습 데이터
- **test.csv**: 예측을 위한 테스트 데이터
- **sample_submission.csv**: 제출 형식 예시

### 전처리된 데이터

- **final_pipe.joblib**: 학습된 전처리 파이프라인과 모델이 포함된 joblib 파일

## 사용 지침

1. Kaggle에서 원본 데이터를 다운로드하여 `data/raw/` 디렉토리에 배치하세요.
2. `src/data.py`의 `load_data()` 함수를 사용하여 데이터를 로드할 수 있습니다.

## 데이터 출처

이 데이터셋은 [Kaggle Prudential Life Insurance Assessment](https://www.kaggle.com/competitions/prudential-life-insurance-assessment/data) 대회에서 가져왔습니다.
