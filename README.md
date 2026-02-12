# 🚀**Spaceship Titanic : 가상 차원 전송 승객 예측 모델 만들기**🚀

**Spaceship Titanic** 경진대회 데이터를 분석하고 머신러닝 모델을 구축하기 위한 프로젝트입니다.  
주요 분석 내용은 `Spaceship_3_수정전.ipynb`, `Spaceship_3_수정후.ipynb`에 포함되어 있습니다.

## 📂 주요 노트북: `Spaceship_3_수정전.ipynb`, `Spaceship_3_수정후.ipynb`

이 노트북은 데이터의 탐색적 분석(EDA)부터 전처리, 파생 변수 생성, 인코딩 및 스케일링, 모델 학습 및 평가까지의 과정을 담고 있습니다.

### 1. 🔍 탐색적 데이터 분석 (EDA)
- **CryoSleep (동면 여부)**: 동면 상태인 승객의 생존율(Transported)이 월등히 높음을 확인.
- **TotalSpending (총 지출)**: 선내 서비스(RoomService, FoodCourt, ShoppingMall, Spa, VRDeck) 지출 합계를 분석. 지출이 없는 승객의 특성 파악.
- **HomePlanet & Destination**: 출신 행성과 목적지 간의 이동 패턴 및 생존율 관계 분석.
- **Age**: 연령대별 생존 분포 확인 및 `AgeGroup` 파생 변수 생성 근거 마련.

### 2. 🛠 데이터 전처리 (Preprocessing) & 결측치 처리 (Imputation)
단순한 평균/최빈값 대체를 넘어, 데이터 간의 상관관계를 활용한 정교한 결측치 처리를 수행했습니다.

- **CryoSleep**: `TotalSpending`이 0원이면 동면 중일 확률이 높으므로 `True`, 지출이 있으면 `False`로 보정.
- **Age**: 전체 승객의 **중앙값(Median)** 으로 대체하여 이상치의 영향 최소화.
- **VIP**: 지출이 0원이거나 미성년자(`Age <= 19`), 또는 `HomePlanet`이 Earth인 경우 `False`일 확률이 높음.
- **Destination**: 승객들이 가장 많이 향하는 `TRAPPIST-1e` (최빈값)로 대체.
- **HomePlanet & Surname**:
    - 같은 `Group`에 속한 승객은 가족/동행일 가능성이 높으므로 고향(`HomePlanet`) 정보를 공유.
    - 성씨(`Surname`)가 같은 승객들의 고향 정보를 참고하여 보간.
- **Cabin**: `Group` 정보를 활용하여 같은 그룹원이면 `Deck`, `Num`, `Side`를 공유하도록 처리.

### 3. ⚙️ 파생 변수 생성 (Feature Engineering)
- **Cabin 분해**: `Deck` / `Num` / `Side` 로 컬럼 분리 (Side: P=Port, S=Starboard).
- **AgeGroup**: 나이를 범주형 구간(Baby, Child, Teenager, Adult, Middle Aged, Senior)으로 그룹화.
- **SpendingGroup**: 지출 금액에 따라 구간화 (0원 구간, 소액, 고액 등).
- **GroupSize / FamilySize**: `PassengerId`와 결합된 그룹/가족 규모 변수 생성.

### 4. 🔢 인코딩 및 스케일링
- **Encoding**: 범주형 변수(`HomePlanet`, `Destination`, `Deck`, `Side` 등)에 대해 **One-Hot Encoding** 적용.
- **Scaling**: 수치형 변수(`Age`, `TotalSpending`, `Num` 등)에 대해 **StandardScaler** 적용하여 모델 학습 효율 증대.

