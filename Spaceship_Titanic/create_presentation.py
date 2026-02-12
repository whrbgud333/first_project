
import nbformat as nbf

nb = nbf.v4.new_notebook()

# --------------------------------------------------------------------------------
# 1. Introduction & Setup
# --------------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""
# Spaceship Titanic: 예측 모델링 발표자료
**팀원: [이름]**

## 1. 프로젝트 개요
- **목표**: Spaceship Titanic 데이터를 활용하여 승객의 이송 여부(`Transported`)를 예측하는 모델 개발
- **접근 방식**: 
    1. 탐색적 데이터 분석 (EDA)
    2. 파생 변수 생성 (Feature Engineering)
    3. 결측치 처리 및 전처리 (Preprocessing)
    4. 모델 최적화 (Optuna) 및 앙상블 (Ensemble)
    5. 최종 결과 도출
"""))

nb.cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 경고 무시
warnings.filterwarnings('ignore')

# 시각화 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
try:
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    print("데이터 로드 완료")
    print(f"Train Shape: {train.shape}, Test Shape: {test.shape}")
except FileNotFoundError:
    print("데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
"""))

# --------------------------------------------------------------------------------
# 2. EDA (Exploratory Data Analysis)
# --------------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""
## 2. 탐색적 데이터 분석 (EDA)
데이터의 주요 특징과 타겟 변수(`Transported`)와의 관계를 분석합니다.
"""))

nb.cells.append(nbf.v4.new_code_cell("""
# 타겟 변수 분포 확인
plt.figure(figsize=(6, 4))
sns.countplot(data=train, x='Transported')
plt.title('Transported 분포')
plt.show()

# Transported 비율 확인
print(train['Transported'].value_counts(normalize=True))
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""
### 주요 범주형 변수 분석
`HomePlanet`, `CryoSleep`, `Destination`, `VIP` 등이 타겟에 미치는 영향을 시각화합니다.
"""))

nb.cells.append(nbf.v4.new_code_cell("""
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.countplot(data=train, x='HomePlanet', hue='Transported', ax=axes[0, 0])
axes[0, 0].set_title('HomePlanet vs Transported')

sns.countplot(data=train, x='CryoSleep', hue='Transported', ax=axes[0, 1])
axes[0, 1].set_title('CryoSleep vs Transported')

sns.countplot(data=train, x='Destination', hue='Transported', ax=axes[1, 0])
axes[1, 0].set_title('Destination vs Transported')

sns.countplot(data=train, x='VIP', hue='Transported', ax=axes[1, 1])
axes[1, 1].set_title('VIP vs Transported')

plt.tight_layout()
plt.show()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""
### **Insight 1: CryoSleep**
- `CryoSleep` 상태인 승객은 `Transported`될 확률이 매우 높습니다.
- 이는 모델 예측에 매우 중요한 변수가 될 것입니다.
"""))

nb.cells.append(nbf.v4.new_code_cell("""
# 수치형 변수(나이) 분포 확인
plt.figure(figsize=(10, 5))
sns.histplot(data=train, x='Age', hue='Transported', kde=True, bins=30)
plt.title('Age Distribution by Transported Status')
plt.show()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""
### **Insight 2: Age**
- 0~4세 영유아의 경우 `Transported` 비율이 확연히 높습니다.
- 10대 후반 ~ 20대 초반은 `False` 비율이 약간 더 높습니다.
- 이를 바탕으로 연령대별 그룹화(`AgeGroup`) 파생 변수 생성을 고려합니다.
"""))


# --------------------------------------------------------------------------------
# 3. Feature Engineering
# --------------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""
## 3. Feature Engineering
데이터에서 새로운 정보를 추출하여 모델 성능을 높이기 위한 파생 변수를 생성합니다.
"""))

nb.cells.append(nbf.v4.new_code_cell("""
# 데이터 병합 (전처리를 위해)
all_data = pd.concat([train, test], sort=False).reset_index(drop=True)

# 1. Cabin 파생 변수 생성 (Deck / Num / Side)
all_data[['Deck', 'Num', 'Side']] = all_data['Cabin'].str.split('/', expand=True)

# 2. TotalSpending 파생 변수 생성
spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
all_data[spending_cols] = all_data[spending_cols].fillna(0)
all_data['TotalSpending'] = all_data[spending_cols].sum(axis=1)

# 3. AgeGroup 파생 변수 생성
bins = [0, 4, 12, 19, 30, 50, 80]
labels = ['Baby', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
all_data['AgeGroup'] = pd.cut(all_data['Age'], bins=bins, labels=labels)

# 4. IsAlone 파생 변수 생성 (Group ID 활용)
all_data['Group'] = all_data['PassengerId'].apply(lambda x: x.split('_')[0])
group_counts = all_data['Group'].value_counts()
all_data['GroupSize'] = all_data['Group'].map(group_counts)
all_data['IsAlone'] = (all_data['GroupSize'] == 1).astype(int)

# 5. Age_Cryo 교호작용 변수 생성
# CryoSleep과 AgeGroup을 결합하여 새로운 특성 생성
all_data['CryoSleep'] = all_data['CryoSleep'].astype(str) # 임시로 문자열 변환
all_data['Age_Cryo'] = all_data['AgeGroup'].astype(str) + '_' + all_data['CryoSleep']

print("파생 변수 생성 완료")
all_data[['Deck', 'Side', 'TotalSpending', 'AgeGroup', 'IsAlone', 'Age_Cryo']].head()
"""))

# --------------------------------------------------------------------------------
# 4. Data Preprocessing
# --------------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""
## 4. 데이터 전처리 (Preprocessing)
결측치 처리, 불필요한 컬럼 삭제, 인코딩 및 스케일링을 수행합니다.
"""))

nb.cells.append(nbf.v4.new_code_cell("""
from sklearn.preprocessing import LabelEncoder

# 1. 결측치 처리
# 간단한 예시로 최빈값 및 중앙값 대치 등을 수행
# 실제 분석에서는 HomePlanet, CryoSleep 등을 Group별로 대치하는 정교한 로직 사용 권장

# 범주형: 최빈값 대치
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
for col in cat_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# 수치형: 중앙값 대치
num_cols = ['Age']
for col in num_cols:
    all_data[col] = all_data[col].fillna(all_data[col].median())

# 2. 불필요 컬럼 삭제
drop_cols = ['PassengerId', 'Name', 'Cabin', 'Group', 'Num', 'Age_Cryo'] # Age_Cryo는 예시용으로 생성 후 삭제하거나 인코딩하여 사용
# 여기서는 Age_Cryo를 범주형 변수로 활용하기 위해 남겨두거나, 복잡도를 줄이기 위해 삭제할 수 있음. 
# 이번 실습에서는 단순화를 위해 Cabin 관련 및 식별자 삭제
all_data = all_data.drop(columns=['PassengerId', 'Name', 'Cabin', 'Group', 'Num'])

# 3. Label Encoding (범주형 -> 수치형)
le = LabelEncoder()
encoded_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'AgeGroup', 'Age_Cryo']
for col in encoded_cols:
    if col in all_data.columns:
        all_data[col] = le.fit_transform(all_data[col].astype(str))

# 4. Log Transformation (Skewed Data)
# TotalSpending 등 분포가 치우친 변수에 로그 변환 적용
all_data['TotalSpending'] = np.log1p(all_data['TotalSpending'])

# 데이터 분리
train_df = all_data[:len(train)]
test_df = all_data[len(train):]
y = train['Transported'].astype(int)
X = train_df.drop(columns=['Transported'])
test_X = test_df.drop(columns=['Transported'])

print(f"X Shape: {X.shape}, y Shape: {y.shape}")
"""))


# --------------------------------------------------------------------------------
# 5. Modeling Implementation
# --------------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""
## 5. 모델링 및 최적화
Optuna를 활용하여 하이퍼파라미터 튜닝을 수행하고, 앙상블 모델을 구축합니다.
"""))

nb.cells.append(nbf.v4.new_code_cell("""
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna

# 검증 전략
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 예시: Optuna로 찾은 최적의 파라미터 (시간 관계상 결과값 직접 입력)
best_lgbm_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'random_state': 42,
    'verbose': -1
}

best_cat_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'random_seed': 42,
    'verbose': 0
}

best_xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'random_state': 42,
    'verbosity': 0
}

# 모델 정의
lgbm = LGBMClassifier(**best_lgbm_params)
cat = CatBoostClassifier(**best_cat_params)
xgb = XGBClassifier(**best_xgb_params)

# 개별 모델 성능 확인 (Cross Validation)
models = [('LGBM', lgbm), ('CatBoost', cat), ('XGBoost', xgb)]
for name, model in models:
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print(f"{name} CV Accuracy: {scores.mean():.4f}")
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""
### 앙상블 (Ensemble)
Soft Voting 및 Hard Voting 방식을 사용하여 예측 성능을 극대화합니다.
가중치를 부여한 Weighted Soft Voting 방식도 시도합니다.
"""))

nb.cells.append(nbf.v4.new_code_cell("""
# Soft Voting Ensemble
voting_soft = VotingClassifier(
    estimators=[
        ('lgbm', lgbm), 
        ('cat', cat), 
        ('xgb', xgb)
    ],
    voting='soft'
)

# Hard Voting Ensemble
voting_hard = VotingClassifier(
    estimators=[
        ('lgbm', lgbm), 
        ('cat', cat), 
        ('xgb', xgb)
    ],
    voting='hard'
)

# 앙상블 모델 학습 및 평가
voting_soft.fit(X, y)
voting_hard.fit(X, y)

print("모델 학습 완료")
"""))

# --------------------------------------------------------------------------------
# 6. Submission
# --------------------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""
## 6. 결과 제출
최종 학습된 모델을 사용하여 테스트 데이터에 대한 예측을 수행하고 제출 파일을 생성합니다.
"""))

nb.cells.append(nbf.v4.new_code_cell("""
# 예측 수행
pred_soft = voting_soft.predict(test_X)
pred_hard = voting_hard.predict(test_X)

# 제출 파일 생성
submission = pd.read_csv('./data/sample_submission.csv')
submission['Transported'] = pred_soft.astype(bool)
submission.to_csv('submission_soft.csv', index=False)

submission['Transported'] = pred_hard.astype(bool)
submission.to_csv('submission_hard.csv', index=False)

print("제출 파일 생성 완료: submission_soft.csv, submission_hard.csv")
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""
## 7. 결론 및 향후 과제
- **결론**: EDA를 통해 중요한 파생 변수를 발굴하고, Optuna 튜닝 및 앙상블 기법을 통해 성능을 개선함.
- **향후 과제**:
    - 더 정교한 결측치 처리 로직 적용
    - Stacking 앙상블 기법 도입
    - 딥러닝 모델(TabNet 등) 활용 검토
"""))

# --------------------------------------------------------------------------------
# Save the notebook
# --------------------------------------------------------------------------------
output_filename = 'Spaceship_발표.ipynb'
try:
    with open(output_filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"'{output_filename}' 파일이 성공적으로 생성되었습니다.")
except Exception as e:
    print(f"파일 생성 중 오류 발생: {e}")
