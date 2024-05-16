# Pandas 

#### Pandas 가 무엇인가? 

Pandas는 데이터 조작 및 분석을 위한 파이썬 라이브러리임. 

주로 테이블 형식의 데이터를 다루는 데 사용되며, 데이터프레임(DataFrame)과 시리즈(Series)라는 두 가지 주요 데이터 구조를 제공한다. 

주요 기능: 
- 데이터프레임(DataFrame):
  - 2차원 데이터 구조로, 행과 열로 구성된 표 형식의 데이터를 다룰 수 있음. 
  - 다양한 데이터 타입(숫자, 문자열, 날짜 등)을 지원하며, 각 열은 서로 다른 데이터 타입을 가질 수 있다. 
- 시리즈(Series):
  - 1차원 데이터 구조로, 인덱스를 가지는 배열과 유사하다. 
  - 단일 열 데이터 또는 리스트 형태의 데이터를 다룰 때 유용하다. 
- 데이터 입출력:
  - CSV, Excel, SQL, JSON, HTML, HDF5 등 다양한 파일 형식의 데이터를 읽고 쓸 수 있는 기능을 제공한다. 

- 데이터 조작 및 변환:
  - 데이터 선택, 필터링, 정렬, 그룹화, 병합, 결합, 피벗 등 다양한 데이터 조작 및 변환 작업을 지원한다. 
  - 결측치 처리, 중복 데이터 제거 등의 데이터 정제 작업도 가능하다. 

- 데이터 분석 및 요약:
  - 데이터의 통계적 요약, 집계, 변환, 시각화 등을 쉽게 수행할 수 있습니다.
  - 데이터 프레임의 행과 열에 대한 다양한 연산을 제공합니다.


#### Q) Pandas 를 통해 데이터 조작이나 변환을 할 수 있다면 데이터 엔지니어링 작업을 할 수 있는거야? Apache Spark 를 대체해서 사용할 수 있는건가?

Pandas는 데이터 조작 및 변환에 매우 유용하며, 데이터 엔지니어링 작업의 많은 부분을 수행할 수 있습니다. 

그러나 Pandas와 Apache Spark는 서로 다른 목적과 사용 사례에 맞게 설계되었기 때문에 직접적인 대체재로 사용되기에는 한계가 있음. 

Panas 는 아무래도 대용량 데이터 처리에 있어 성능이 떨어질 수 있기 떄문임. 

Apache Spark 는 대규모 데이터를 효율적으로 처리할 수 있음. 

PySpark 를 이용하면 Pandas 와 Spark 의 장점을 모두 취할 수 있다. 


#### 머신러닝과 데이터 엔지니어링에서 주로 사용하는 Pandas 의 대표적인 연산들은?

1. 데이터 로드 및 저장
- 데이터 로드
  - CSV 파일 읽기: pd.read_csv()
  - Excel 파일 읽기: pd.read_excel()
  - SQL 데이터베이스에서 읽기: pd.read_sql()

```python
import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('data.csv')

# Excel 파일 읽기
df = pd.read_excel('data.xlsx')

# SQL 데이터베이스에서 읽기
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table_name', conn)
```

- 데이터 저장
  - CSV 파일로 저장: df.to_csv()
  - Excel 파일로 저장: df.to_excel()
  - SQL 데이터베이스에 저장: df.to_sql()

```python
# CSV 파일로 저장
df.to_csv('data.csv', index=False)

# Excel 파일로 저장
df.to_excel('data.xlsx', index=False)

# SQL 데이터베이스에 저장
df.to_sql('table_name', conn, if_exists='replace', index=False)
```

2. 데이터 확인 및 탐색
- 데이터 구조 확인: df.head(), df.tail(), df.shape, df.info()
- 기술 통계: df.describe()

```python
# 데이터 구조 확인
print(df.head())
print(df.tail())
print(df.shape)
print(df.info())

# 기술 통계
print(df.describe())
```

3. 데이터 선택 및 필터링
- 열 선택: df['column_name'], df[['col1', 'col2']]
- 행 선택: df.iloc[], df.loc[]
- 조건 필터링: df[df['column_name'] > value]

```python
# 열 선택
print(df['column_name'])
print(df[['col1', 'col2']])

# 행 선택 (인덱스로 선택)
print(df.iloc[0:5])

# 행 선택 (라벨로 선택)
print(df.loc[df['column_name'] > value])
```

4. 데이터 정제 및 변환
- 결측치 처리: df.isnull(), df.dropna(), df.fillna()
- 중복 데이터 처리: df.duplicated(), df.drop_duplicates()
- 데이터 타입 변환: df['column_name'].astype()

```python
# 결측치 처리
print(df.isnull().sum())
df = df.dropna()
df = df.fillna(0)

# 중복 데이터 처리
print(df.duplicated().sum())
df = df.drop_duplicates()

# 데이터 타입 변환
df['column_name'] = df['column_name'].astype('int')
```

5. 데이터 조작
- 새로운 열 추가: df['new_column'] = values
- 열 이름 변경: df.rename()
- 열 삭제: df.drop()
- 데이터 병합: pd.concat(), pd.merge()

```python
# 새로운 열 추가
df['new_column'] = df['existing_column'] * 2

# 열 이름 변경
df = df.rename(columns={'old_name': 'new_name'})

# 열 삭제
df = df.drop(columns=['column_name'])

# 데이터 병합
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})
merged_df = pd.merge(df1, df2, on='key', how='inner')
```

6. 데이터 그룹화 및 집계
- 그룹화: df.groupby()
- 집계 함수: sum(), mean(), count(), agg()

```python
# 그룹화 및 집계
grouped_df = df.groupby('column_name')
print(grouped_df.sum())
print(grouped_df.mean())
print(grouped_df.size())

# 여러 집계 함수 적용
grouped_df = df.groupby('column_name').agg({'col1': 'sum', 'col2': 'mean'})
print(grouped_df)
```

7. 피벗 및 재구조화
- 피벗 테이블: pd.pivot_table()
- 데이터 재구조화: df.melt(), df.pivot()

```python
# 피벗 테이블
pivot_df = pd.pivot_table(df, values='value', index='index_col', columns='column_col', aggfunc='mean')
print(pivot_df)

# 데이터 재구조화
melted_df = pd.melt(df, id_vars=['id_col'], value_vars=['col1', 'col2'])
print(melted_df)

pivoted_df = melted_df.pivot(index='id_col', columns='variable', values='value')
print(pivoted_df)
```

8. 시각화
- 기본 시각화: df.plot()
- Seaborn, Matplotlib과 통합: sns.scatterplot(), plt.plot()

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 기본 시각화
df.plot(kind='line', x='col1', y='col2')
plt.show()

# Seaborn 시각화
sns.scatterplot(data=df, x='col1', y='col2')
plt.show()
```




#### Pandas 의 대표적인 예제들에 대해서 

1. 가상 데이터 생성

```python
import pandas as pd
import numpy as np

# 랜덤 데이터 생성
np.random.seed(42)
n_samples = 1000

data = {
    'PassengerId': np.arange(1, n_samples + 1),
    'Survived': np.random.randint(0, 2, size=n_samples),
    'Pclass': np.random.randint(1, 4, size=n_samples),
    'Name': ['Name' + str(i) for i in range(n_samples)],
    'Sex': np.random.choice(['male', 'female'], size=n_samples),
    'Age': np.random.uniform(1, 80, size=n_samples),
    'SibSp': np.random.randint(0, 6, size=n_samples),
    'Parch': np.random.randint(0, 6, size=n_samples),
    'Ticket': ['Ticket' + str(i) for i in range(n_samples)],
    'Fare': np.random.uniform(10, 300, size=n_samples),
    'Cabin': ['Cabin' + str(i) if np.random.rand() > 0.7 else np.nan for i in range(n_samples)],
    'Embarked': np.random.choice(['C', 'Q', 'S'], size=n_samples)
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# 데이터 확인
print(df.head())
print(df.info())
print(df.describe())
```

2. 데이터 탐색

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 생존자 분포 확인
sns.countplot(x='Survived', data=df)
plt.show()

# 성별에 따른 생존자 분포
sns.countplot(x='Survived', hue='Sex', data=df)
plt.show()

# 연령대에 따른 생존자 분포
sns.histplot(df, x='Age', hue='Survived', bins=30, kde=True)
plt.show()
```

3. 데이터 전처리

```python
# 결측치 처리
# df['Age'].fillna(df['Age'].median(), inplace=True): Age 열의 결측치(NaN)를 Age 열의 중앙값(median)으로 채운다. inplace=True는 원본 데이터프레임(df)을 수정함을 의미한다.
# df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True): Embarked 열의 결측치를 Embarked 열의 최빈값(mode)으로 채운다. mode()[0]은 최빈값이 여러 개일 경우 첫 번째 최빈값을 사용한다. 
# df['Cabin'].fillna('Unknown', inplace=True): Cabin 열의 결측치를 'Unknown'이라는 문자열로 채운다. 
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin'].fillna('Unknown', inplace=True)

# 범주형 데이터 인코딩
# df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}): Sex 열의 값을 'male'은 0으로, 'female'은 1로 매핑하여 숫자로 변환한다. 
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# df = pd.get_dummies(df, columns=['Embarked'], drop_first=True): Embarked 열을 원-핫 인코딩(one-hot encoding)한다. 원-핫 인코딩은 범주형 변수를 여러 개의 이진 변수로 변환한다. drop_first=True는 첫 번째 더미 변수를 제거하여 다중공선성 문제를 방지한다. 
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 불필요한 열 제거
# df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True): PassengerId, Name, Ticket, Cabin 열을 데이터프레임에서 삭제한다. axis=1은 열(column)을 삭제함을 의미하며, inplace=True는 원본 데이터프레임을 수정한다.
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 데이터 확인
print(df.head())
print(df.info())
```

4. 머신러닝 모델 훈련

```python
# from sklearn.model_selection import train_test_split: train_test_split 함수를 임포트하여 데이터셋을 학습용과 테스트용으로 분할할 수 있게 한다. 
from sklearn.model_selection import train_test_split

# from sklearn.ensemble import RandomForestClassifier: 랜덤 포레스트 분류기를 임포트하여 분류 모델을 생성할 수 있게 한다. 
from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import accuracy_score: 모델의 성능을 평가하기 위해 정확도(accuracy)를 계산하는 함수를 임포트한다. 
from sklearn.metrics import accuracy_score

# 특징과 레이블 분리
# X = df.drop('Survived', axis=1): 데이터프레임 df에서 'Survived' 열을 제외한 나머지 열들을 특징(feature) 변수로 선택한다. axis=1은 열을 의미한다. 
X = df.drop('Survived', axis=1)

# y = df['Survived']: 'Survived' 열을 레이블(label) 변수로 선택한다. 
y = df['Survived']

# 학습 데이터와 테스트 데이터 분리
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42): train_test_split 함수를 사용하여 특징 변수 X와 레이블 변수 y를 학습용 데이터와 테스트용 데이터로 분리한다. 
# test_size=0.2는 전체 데이터의 20%를 테스트 데이터로 할당하고, random_state=42는 결과를 재현할 수 있도록 랜덤 시드(seed)를 설정한다. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 훈련
# model = RandomForestClassifier(n_estimators=100, random_state=42): 100개의 결정 트리(decision tree)로 구성된 랜덤 포레스트 분류기 모델을 생성한다. 
# random_state=42는 모델의 결과를 재현할 수 있도록 랜덤 시드(seed)를 설정한다. 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
# y_pred = model.predict(X_test): 훈련된 모델을 사용하여 테스트 데이터(X_test)에 대한 예측값을 생성한다. 
y_pred = model.predict(X_test)

# 정확도 평가
# accuracy_score(y_test, y_pred): 실제 레이블(y_test)과 모델이 예측한 값(y_pred)을 비교하여 정확도(accuracy)를 계산한다. 
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```


#### Q) Numpy 라이브러리란?

파이썬에서 과학적 계산을 위한 라이브러리로, 대규모 다차원 배열과 행렬 연산을 효율적으로 수행할 수 있는 기능을 제공한다. 

또한, 수학 함수 라이브러리도 포함되어 있어 배열을 처리하는 데 매우 유용하다. 


#### Q) 원 핫 인코딩이란? 그리고 map 을 써서 범주형 변수를 이진 변수로 변환하는 방법과 get_dummies() 함수를 사용하는 이유는 뭔데?

map() 메서드를 사용하면 특정 열의 값들을 매핑하여 다른 값들로 변환할 수 있다. 

이 방법은 주로 범주형 변수에 두 가지 값만 있을 때 사용된다. 예를 들어, 'Sex' 열에 'male'과 'female' 두 가지 값만 있을 경우, 이를 이진 변수로 변환할 수 있다. 

```python
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
```

get_dummies() 함수를 사용하는 이유는 pd.get_dummies() 함수는 범주형 변수를 원-핫 인코딩하는 데 사용된다. 이 함수는 범주형 변수의 각 고유 값에 대해 새로운 이진 열을 생성ㅎ나다.  여러 개의 범주를 가진 변수를 이진 벡터로 변환하는 데 유용함.

원-핫 인코딩은 범주형 데이터를 이진 벡터로 변환하는 기법임. 

각 범주를 하나의 이진 벡터로 표현하며, 벡터의 길이는 범주의 총 개수와 같고, 특정 범주에 해당하는 위치는 1로, 나머지 위치는 0으로 표시된다. 

예를 들어, Embarked 열의 값이 ['C', 'Q', 'S']인 경우:
- 'C'는 [1, 0, 0]
- 'Q'는 [0, 1, 0]
- 'S'는 [0, 0, 1]

#### Q) sklearn 라이브러리는 뭔데?

sklearn은 Scikit-Learn의 약칭으로, 파이썬에서 가장 널리 사용되는 머신러닝 라이브러리 중 하나임. 

Scikit-Learn은 다양한 머신러닝 알고리즘을 쉽고 일관된 인터페이스로 제공하여 데이터 분석과 예측 모델링 작업을 간편하게 할 수 있게 한다. 


#### Q) 머신러닝 알고리즘을 사용하고 싶다면 Pytorch 나 Tensorflow 를 사용하는게 아니라 sklearn 라이브러리를 쓰는거야?

모두 머신러닝 및 딥러닝 작업에 사용되는 인기 있는 라이브러리임. 

각각의 라이브러리는 특정 용도에 더 적합하며, 사용자의 필요에 따라 선택될거다. 

Scikit-Learn: 전통적인 머신러닝 알고리즘과 데이터 전처리에 적합하다. 빠르게 프로토타입을 만들고 간단한 모델을 평가하는 데 유리함. 

PyTorch: 유연한 딥러닝 모델을 연구하고 개발할 때 적합하다. 특히 동적 계산 그래프를 필요로 하는 연구 환경에 유리함. 

TensorFlow: 대규모 배포와 엔터프라이즈 솔루션 딥러닝에 적합합니다. 정적 계산 그래프와 확장성 있는 배포가 필요할 때 유리함. 





