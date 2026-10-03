# 🎓 학습자 수료 예측 AI — DACON x BDA 제2회

![Rank](https://img.shields.io/badge/🏆%20Rank-Top%204%25-gold)
[![DACON](https://img.shields.io/badge/DACON-월간%20데이콘-0C4DA2)](https://dacon.io/competitions/official/236664/overview/description)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00)
![Optuna](https://img.shields.io/badge/Optuna-HPO-4B8BBE)
![Metric](https://img.shields.io/badge/Metric-F1%20Score-555)

> BDA(빅데이터분석학회) 9기 학습자의 설문 데이터로 **10기 학습자의 수료 여부(`completed`)** 를 예측하는 이진 분류 대회
> 기간: 2026.01.12 ~ 2026.02.23 · 참가자 1,257명 · 정형 데이터 · 평가지표 F1 Score

## 📊 결과

| 구분 | 점수 |
| --- | --- |
| **최종 순위** | **상위 4%** (참가자 1,257명) |
| OOF F1 (3-Fold, 5-seed 앙상블) | 약 0.51 |
| **Public LB 최고 점수** | **0.4421** |

## 🔍 문제 이해

- 학습 데이터 748명, 테스트 814명 · 수료 비율 약 **30%** 로 클래스 불균형
- 피처 대부분이 **설문 응답(범주형/자유 텍스트)** 이라 전처리와 그룹화가 핵심
- F1이 지표라서 확률 예측 후 **threshold를 얼마로 자르느냐** 가 점수에 큰 영향

## 🛠 접근 방법

### 1. 피처 엔지니어링
설문 원문을 모델이 쓸 수 있는 형태로 정리했습니다.

| 피처 | 설명 |
| --- | --- |
| `inflow_route` 그룹화 | 유입 경로를 커뮤니티 / SNS / 지인 / 내부 네트워크 / 대외활동 플랫폼 등으로 묶음 |
| `major1_1` 그룹화 | 키워드 기반으로 전공을 IT·경영·자연과학·사회과학 등 대분류로 매핑 |
| `time_band`, `time_risk` | 주당 투자 가능 시간을 구간화 |
| `cert_cnt`, `cert_weighted` | 자격증 목록 정제 후 개수 및 가중 점수 (SQLD·빅데이터분석기사 등) |
| `topic_*`, `topic_count` | 원데이 클래스 희망 주제 멀티핫 인코딩 |
| `sincerity_score` | 자유 서술 문항(지원 동기, 얻고 싶은 것 등)의 답변 길이 합 → 성실도 지표 |
| `is_experienced` | 재등록 여부 + 이전 기수 수강 이력 |
| `senior_urgency` | 이수 학기 × 투자 시간 |
| `goal_alignment_score` | 희망 직무와 희망 학습 주제의 키워드 일치도 |

### 2. 피처 선택
- CatBoost `select_features` (**RecursiveByShapValues**)로 k = 5 ~ 24개 구간을 탐색
- 누수를 막으려고 피처 선택용 데이터를 따로 떼어 놓고 진행
- 최종적으로 **22개 피처** 사용

### 3. 모델링
- **CatBoostClassifier** (범주형 변수 바로 처리)
- **Optuna (TPE, 50 trials)** 로 `learning_rate`, `depth`, `l2_leaf_reg`, `min_data_in_leaf`, `random_strength` 튜닝
- 목적 함수: 3-Fold OOF 예측에서 threshold를 최적화한 F1

```python
CatBoostClassifier(
    learning_rate=0.0987, depth=7, l2_leaf_reg=3.84,
    min_data_in_leaf=35, random_strength=5.19,
    iterations=2000, early_stopping_rounds=200, eval_metric="AUC",
)
```

### 4. 앙상블 & Threshold 튜닝
- **Multi-seed(5개) × 3-Fold** 평균으로 예측 분산 감소
- OOF 기준 threshold를 0.002 단위로 탐색하고, fold별 F1과 양성 비율이 안정적인지 확인
- threshold를 조금씩 바꾸며 이전 제출과 예측이 얼마나 다른지(불일치 개수)와 LB 점수를 비교

| threshold | Public LB |
| --- | --- |
| 0.276 | 0.4318 |
| 0.279 | 0.4383 |
| 0.280 | 0.4365 |
| 0.2948 | 0.4315 |
| **0.293** | **0.4421** |

## 💡 배운 점
- 앙상블로 OOF F1은 소폭 오르는 데 그쳤지만(+0.001), **예측이 안정**되어 threshold를 정하기 쉬워짐
- 양성 비율이 30%인 데이터에서 F1을 높이려면 threshold를 낮춰 recall을 확보하는 쪽이 유리했음 (recall 0.90 / precision 0.33)
- OOF와 LB 사이 차이가 커서, **threshold 미세 조정은 LB에 과적합될 위험**이 있다는 점을 체감

## 📁 구조
```
.
├── notebooks/
│   └── catboost_completion_prediction.ipynb   # 전처리 ~ 제출 파일 생성 전체 과정
├── requirements.txt
└── README.md
```

## ▶️ 실행 방법
1. [대회 페이지](https://dacon.io/competitions/official/236664/data)에서 데이터를 받아 `train.csv`, `test.csv`, `sample_submission.csv`를 준비
2. 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```
3. 노트북의 데이터 경로(`os.chdir(...)`)를 내 환경에 맞게 바꾼 뒤 실행 (Google Colab 기준으로 작성)

> ⚠️ 대회 규정에 따라 원본 데이터는 저장소에 포함하지 않았습니다.
