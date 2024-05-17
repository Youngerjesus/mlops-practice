# Dask 

Dask는 파이썬 생태계에서 데이터 병렬 처리를 위한 라이브러리임. 

빅 데이터 처리를 쉽게 하고, 계산을 병렬화하며, 메모리 효율성을 높이는 데 유용함. 

Dask는 pandas와 유사한 API를 제공하여 데이터 분석을 병렬로 수행할 수 있도록 도와주며, NumPy, pandas, scikit-learn 등의 기존 파이썬 라이브러리와 원활하게 통합됨. 

Dask 주요 특징: 
- 병렬 컴퓨팅: Dask는 여러 CPU 코어를 활용하여 데이터 처리 속도를 높일 수 있음. 
- 스케줄러: Dask에는 로컬 스레드 및 프로세스를 관리하는 간단한 스케줄러와, 클러스터에서 대규모로 작업을 관리할 수 있는 분산 스케줄러가 포함되어 있음. 
- 대용량 데이터 처리: 메모리에 맞지 않는 대용량 데이터를 처리할 수 있음. 
- 유연한 아키텍처: 사용자 정의 작업 흐름을 쉽게 만들 수 있으며, 다양한 작업 유형을 지원함. 

주요 컴포넌트: 
- Dask Arrays: NumPy 배열과 유사하지만, 매우 큰 배열을 여러 작은 블록으로 나누어 병렬로 처리할 수 있음. 
- Dask DataFrames: pandas DataFrame과 유사하게 동작하며, 큰 데이터셋을 여러 파티션으로 나누어 병렬로 작업할 수 있음. 
- Dask Bags: 비정형 데이터나 매우 큰 리스트 같은 데이터를 처리하기 위한 도구임. 
- Dask Delayed: 일반 파이썬 함수들을 병렬로 실행할 수 있도록 도와줌. 

Dask를 사용하면 기존의 pandas 코드나 NumPy 코드를 거의 변경하지 않고도 병렬 처리를 활용하여 성능을 크게 향상시킬 수 있음. 

#### Q) Dask 는 PySpark 안에서 주로 사용해? 

Dask와 PySpark는 모두 빅 데이터 처리를 위한 도구이지만, 이들은 서로 다른 생태계에서 사용됨. 

Dask: 로컬 머신이나 소규모 클러스터에서 좋은 성능을 발휘함.

PySpark 는 주로 대규모 클러스터 환경에서 사용하고. 

#### Q) Dask 도 분산 환경에서 사용가능해? 

Dask는 소규모의 로컬 환경뿐만 아니라 분산 클러스터에서도 효율적으로 작동할 수 있도록 설계되었음. 

Dask 분산 환경 사용 방법: 
- 1. Dask Scheduler와 Worker 설정:
  - Dask는 중앙 스케줄러와 여러 워커로 구성된 분산 환경을 지원한다. 중앙 스케줄러는 작업을 관리하고, 워커들은 실제 계산을 수행한다.
  - Dask의 분산 클러스터를 설정하려면 dask-scheduler와 dask-worker 명령어를 사용하여 각각 스케줄러와 워커를 실행할 수 있다. 

- 2. Dask.distributed 모듈:
  - Dask의 distributed 모듈은 분산 환경에서 작업을 수행할 수 있는 기능을 제공한다. 이를 통해 로컬 머신에서 클러스터에 연결하거나 클라우드 환경에서도 Dask를 사용할 수 있다. 
  - 예를 들어, 다음과 같이 클러스터에 연결할 수 있음. 


```python
from dask.distributed import Client

client = Client('scheduler-address:port')
```

- 3. 자동 스케일링:

  - Dask는 워커 수를 자동으로 조정하여 작업 부하에 맞게 클러스터를 확장하거나 축소할 수 있음. 이는 클라우드 환경에서 특히 유용함.
  - Kubernetes나 YARN 같은 오케스트레이션 도구와 통합하여 자동 스케일링을 구현할 수도 있음. 


- Dask의 주요 분산 기능: 
  - Task Graphs: 
    - Dask는 작업을 DAG(Directed Acyclic Graph) 형태로 표현하여, 각 작업 간의 의존성을 관리하고 최적화된 병렬 처리를 수행할 수 있음. 
  - Fault Tolerance:
    - Dask는 작업이 실패했을 때 자동으로 재시도하거나, 실패한 작업을 감지하여 복구할 수 있는 기능을 제공함. 이는 분산 환경에서 높은 신뢰성을 보장한다. 
  - Data Management:
    - Dask는 데이터를 분할하여 여러 워커에 분산시키고, 각 워커가 독립적으로 데이터를 처리할 수 있도록 함. 이는 대규모 데이터셋을 효율적으로 처리하는 데 도움이 됨.
  - Scheduling:
    - Dask의 스케줄러는 작업을 최적화하여 워커 간의 작업 부하를 균등하게 분배함. 이는 작업 성능을 최적화하고 리소스를 효율적으로 사용할 수 있도록 한다. 

```python
from dask.distributed import Client
import dask.array as da

# Dask 클러스터에 연결
client = Client('scheduler-address:port')

# 큰 배열 생성
x = da.random.random((10000, 10000), chunks=(1000, 1000))

# 배열 연산
result = x.mean().compute()

print(result)
```

#### 예시: 쿠버네티스 환경에서 Dask 를 이용해서 분산 데이터 처리를 하는 예제

쿠버네티스 클러스터가 설정되어 있다는 가정하에. 

1. Dask Helm 차트를 이용한 배포: 

Helm 저장소 추가 및 업데이트:

```shell 
helm repo add dask https://helm.dask.org/
helm repo update
```

Dask 배포:
- 기본 설정으로 Dask 스케줄러와 워커를 배포. 필요에 따라 values.yaml 파일을 사용하여 커스텀할 수 있음. 

```shell
helm install dask dask/dask
```

2. 클라이언트 설정

Dask 클라이언트를 설정하여 배포된 Dask 클러스터에 연결하는 것. 

Dask 클라이언트로 작업을 제출하면 Helm 으로 배포한 Dask 워커와 스케줄러가 알아서 실행하는거임.

Dask 클라이언트 파이썬으로 작성.

일반적으로는 Dask 클라이언트를 로컬이나 클라우드 환경에서 작성해서 잡을 제출하는 식으로 구현됨. 

```python
from dask.distributed import Client
import dask.array as da

# Dask 클라이언트 생성 (포트 포워딩을 통해 로컬에서 연결)
client = Client('tcp://localhost:8786')

# 큰 배열 생성
x = da.random.random((10000, 10000), chunks=(1000, 1000))

# 배열 연산
result = x.mean().compute()

print(result)
```


#### Dask 에제: 

```python
import time
import pandas as pd
import numpy as np

#  dask는 병렬 컴퓨팅을 위해 설계된 라이브러리로, dask.dataframe은 pandas의 데이터프레임을 분산 환경에서 처리할 수 있도록 확장한 것 
import dask.dataframe as dd


# create_datasets라는 함수를 정의합니다. 이 함수는 두 개의 정수 인자 nrows와 ncols를 받아들이며, 두 개의 pandas 데이터프레임을 반환함. 
def create_datasets(nrows: int, ncols: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    # 딕셔너리 컴프리헨션을 사용하여 main_data라는 딕셔너리를 만듬. 키는 "col_i" 형식의 문자열이고, 값은 nrows 길이의 무작위 숫자 배열입니다. ncols만큼의 컬럼이 생성됨. 
    main_data = {f"col_{i}": np.random.rand(nrows) for i in range(ncols)}
    
    # 또 다른 딕셔너리 컴프리헨션을 사용하여 ref_data를 만듬. 키는 "col_i" 형식의 문자열이고, 값은 nrows를 10으로 나눈 길이의 무작위 숫자 배열임. ncols만큼의 컬럼이 생성된다. 
    ref_data = {f"col_{i}": np.random.rand(nrows // 10) for i in range(ncols)}
    
    # main_data 딕셔너리를 사용하여 main_df라는 pandas 데이터프레임을 생성함. 
    main_df = pd.DataFrame(main_data)
    
    # ref_data 딕셔너리를 사용하여 ref_df라는 pandas 데이터프레임을 생성함. 
    ref_df = pd.DataFrame(ref_data)
    
    # main_df와 ref_df 데이터프레임을 반환함. 
    return main_df, ref_df

# pandas_operations라는 함수를 정의한다. 이 함수는 두 개의 pandas 데이터프레임을 인자로 받아들이며, 두 개의 실행 시간을 반환한다. 
def pandas_operations(main_df: pd.DataFrame, ref_df: pd.DataFrame) -> tuple[float, float]:
    # 현재 시간을 초 단위로 기록하여 start_time_agg에 저장함. 이는 그룹화 및 집계 작업의 시작 시간을 나타냄. 
    start_time_agg = time.time()
    
    # main_df 데이터프레임을 "col_0" 컬럼을 기준으로 그룹화하고, 각 그룹의 평균을 계산하여 grouped에 저장한다. 
    grouped = main_df.groupby("col_0").mean()
    
    # 현재 시간을 초 단위로 기록하여 end_time_agg에 저장한다. 이는 그룹화 및 집계 작업의 종료 시간을 나타냄. 
    end_time_agg = time.time()

    # 현재 시간을 초 단위로 기록하여 start_time_join에 저장함. 이는 조인 작업의 시작 시간을 나타냄 
    start_time_join = time.time()
    
    # main_df와 ref_df를 "col_0" 컬럼을 기준으로 왼쪽 조인(Left Join)하여 joined 데이터프레임을 생성함. 
    joined = main_df.merge(ref_df, on="col_0", how="left")
    
    # 현재 시간을 초 단위로 기록하여 end_time_join에 저장합니다. 이는 조인 작업의 종료 시간을 나타냄. 
    end_time_join = time.time()

    # 그룹화 및 집계 작업과 조인 작업에 소요된 시간을 각각 반환함. 
    # end_time_agg - start_time_agg는 그룹화 및 집계 작업의 실행 시간을, 
    # end_time_join - start_time_join은 조인 작업의 실행 시간을 나타낸다. 
    return end_time_agg - start_time_agg, end_time_join - start_time_join


# 이 함수는 main_df와 ref_df라는 두 개의 pandas 데이터프레임과 npartitions라는 정수를 입력으로 받음. 
# 반환값은 두 개의 float로, 각각 그룹화 및 병합 작업의 실행 시간을 나타낸다. 
def dask_operations(
    main_df: pd.DataFrame, ref_df: pd.DataFrame, npartitions: int
) -> tuple[float, float]:
  
    # main_df를 Dask 데이터프레임으로 변환함. 
    # npartitions는 데이터를 몇 개의 파티션으로 나눌지를 결정한다. 
    # dmain_df는 Dask 데이터프레임임. 
    dmain_df = dd.from_pandas(main_df, npartitions=npartitions)
    
    # ref_df를 Dask 데이터프레임으로 변환함.
    # npartitions를 사용하여 데이터를 나눔. 
    # dref_df는 Dask 데이터프레임임. 
    dref_df = dd.from_pandas(ref_df, npartitions=npartitions)

    # 그룹화 작업의 시작 시간을 기록함. 
    start_time_agg = time.time()
    grouped_task = dmain_df.groupby("col_0").mean()
    grouped = grouped_task.compute()
    end_time_agg = time.time()
    grouped_task.visualize("grouped.svg")

    start_time_join = time.time()
    joined_task = dmain_df.merge(dref_df, on="col_0", how="left")
    joined = joined_task.compute()
    end_time_join = time.time()
    joined_task.visualize("joined.svg")

    return end_time_agg - start_time_agg, end_time_join - start_time_join

main_df, ref_df = create_datasets(10_000_000, 5)

pandas_agg_time, pandas_join_time = pandas_operations(main_df, ref_df)
dask_agg_time, dask_join_time = dask_operations(main_df, ref_df, npartitions=10)

print("Pandas 집계 시간:", pandas_agg_time, "초")
print("Pandas 조인 시간:", pandas_join_time, "초")
print("Dask 집계 시간:", dask_agg_time, "초")
print("Dask 조인 시간:", dask_join_time, "초")
```

#### Q) main_data 표현은? 

```python
{
    'col_0': array([0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ]),
    'col_1': array([0.64589411, 0.43758721, 0.891773  , 0.96366276, 0.38344152]),
    'col_2': array([0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606])
}
```


#### Q) 파이썬에서 딕셔녀러 데이터와 딕셔너리 컴프리헨션이란?

딕셔너리는 키(key)와 값(value)의 쌍으로 이루어진 데이터 구조임. 각 키는 고유하며, 값을 참조하는 데 사용됩니다. 딕셔너리는 중괄호 {}로 정의됨.

```python
# 빈 딕셔너리 생성
my_dict = {}

# 키-값 쌍을 포함한 딕셔너리 생성
my_dict = {
    'name': 'Alice',
    'age': 30,
    'city': 'New York'
}
```

딕셔너리 요소 접근
```python
# 특정 키의 값 접근
print(my_dict['name'])  # 출력: Alice

# get() 메서드로 값 접근 (키가 없는 경우 None 반환)
print(my_dict.get('age'))  # 출력: 30
print(my_dict.get('address'))  # 출력: None
```

딕셔너리 요소 추가 및 수정
```python
# 새로운 키-값 쌍 추가
my_dict['address'] = '123 Main St'

# 기존 키의 값 수정
my_dict['age'] = 31
```

딕셔너리 요소 삭제

```python
# 특정 키-값 쌍 삭제
del my_dict['city']

# pop() 메서드로 값 삭제
address = my_dict.pop('address')
```

딕셔너리 컴프리헨션은 반복문을 사용하여 딕셔너리를 간결하게 생성하는 방법임. 

리스트 컴프리헨션과 유사하지만, 중괄호 {}를 사용하여 딕셔너리를 생성함. 

```python
# {key_expression: value_expression for item in iterable}
squares = {x: x*x for x in range(6)}

print(squares)  # 출력: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
```

조건부 딕셔너리 컴프리헨션

```python
# {key_expression: value_expression for item in iterable if condition}
even_squares = {x: x*x for x in range(6) if x % 2 == 0}

print(even_squares)  # 출력: {0: 0, 2: 4, 4: 16}
```

#### Q) main_df 는 어떤식으로 데이터가 표현되는가? 

이차원 테이블로, 기존 key 값은 테이블의 '열' 로 들어가서 표현됨. 

이렇게 표현될 것. 

![](./images/dask%20dataframe.png)
