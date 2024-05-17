# Apache Arrow 

Apache Arrow는 컬럼 방식의 데이터 포맷 및 데이터 처리 엔진을 제공하는 오픈 소스 프로젝트임. 

주요 특징은 다음과 같다: 
- Arrow는 컬럼 지향 포맷을 사용하여 데이터 처리와 분석을 최적화한다. 이는 행 기반 포맷보다 메모리 사용과 I/O 효율성을 크게 향상시킴. 
- Arrow는 여러 프로그래밍 언어(C++, Java, Python, R 등)에서 사용될 수 있도록 설계됨. 동일한 데이터 포맷을 사용함으로써 언어 간 데이터 이동이 효율적임
- Arrow는 메모리 내 데이터 구조를 표준화하여, 데이터 복사 없이도 여러 프로세스와 스레드에서 데이터를 공유할 수 있음. 
- Arrow의 데이터 구조는 CPU 캐시 친화적으로 설계되어 고속 벡터화 연산아 가능함. 
- Arrow는 효율적인 데이터 직렬화 및 역직렬화 메커니즘을 제공하여, 네트워크나 디스크를 통한 데이터 전송 속도를 높일 수 있음.

Apache Arrow 와 Dask 비교: 
- 겹치는 부분: 
  - 둘 다 대규모 데이터 처리를 위해 사용될 수 있으며, 높은 성능을 목표로 함. 
- 차이점: 
  - Arrow는 메모리 내 데이터 형식을 최적화하여 언어 간 호환성과 고속 데이터 전송에 중점을 둔다. Dask는 여러 파티션으로 나누어 병렬 처리할 수 있는 데이터 프레임 및 배열을 제공함.
  - Dask는 클러스터 환경에서 분산 컴퓨팅을 지원하며, 여러 머신에 걸쳐 작업을 수행할 수 있다. Arrow는 주로 단일 노드 내에서 고성능 처리를 최적화함. 
  - Arrow는 고성능 데이터 전송, 언어 간 데이터 공유 및 벡터화 연산이 중요한 경우에 적합하다  
  - Dask는 메모리를 초과하는 대규모 데이터 세트 처리와 클러스터 환경에서의 분산 컴퓨팅 작업에 적함함. 

# Matplotlib 

Python에서 데이터를 시각화하기 위한 라이브러리임. 

다양한 유형의 플롯과 그래프를 생성할 수 있음.

주요 특징: 
- 선 그래프, 막대 그래프, 산점도, 히스토그램, 파이 차트 등 다양한 유형의 그래프를 지원함. 
- 그래프의 축, 레이블, 제목, 색상, 스타일 등을 세부적으로 설정할 수 있음. 
- 그래프를 PNG, PDF, SVG 등 다양한 형식으로 저장할 수 있음. 
- 주피터 노트북(Jupyter Notebook)과 같은 인터랙티브 환경에서 실시간으로 그래프를 조작하고 시각화할 수 있음. 
- Pandas, NumPy 등 다른 데이터 분석 라이브러리와 쉽게 통합하여 사용할 수 이씅ㅁ. 

예제: 선그래프 

```python
import matplotlib.pyplot as plt

# 데이터 생성
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# 그래프 생성
plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot Example')
plt.show()
```

예제: 막대 그래프 (Bar Chart)
```python
import matplotlib.pyplot as plt

# 데이터 생성
categories = ['A', 'B', 'C', 'D']
values = [3, 7, 5, 2]

# 그래프 생성
plt.bar(categories, values)
plt.xlabel('Category')
plt.ylabel('Values')
plt.title('Bar Chart Example')
plt.show()
```

#### Q) Matplotlib 으로 그래프를 그려낼 수 있는데 어떤 상황에 어떤 그래프를 그려내면 좋은지 정리해줘.

1. 선 그래프 (Line Plot)

사용 상황:
- 시간에 따른 데이터 변화를 나타낼 때
- 두 변수 간의 관계를 시각화할 때
- 연속적인 데이터를 시각화할 때

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

plt.plot(x, y)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Line Plot Example')
plt.show()
```

![](./images/Line%20Plot.png)

2. 막대 그래프 (Bar Chart)

사용 상황:
- 카테고리별 데이터를 비교할 때
- 수량을 비교할 때
- 빈도를 나타낼 때


```python
categories = ['A', 'B', 'C', 'D']
values = [3, 7, 5, 2]

plt.bar(categories, values)
plt.xlabel('Category')
plt.ylabel('Values')
plt.title('Bar Chart Example')
plt.show()
```

![](./images/Bar%20Plot.png)

3. 산점도 (Scatter Plot)

사용 상황:
- 두 변수 간의 관계를 탐색할 때
- 데이터의 분포를 확인할 때
- 이상치(outlier)를 식별할 때

```python
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

plt.scatter(x, y)
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('Scatter Plot Example')
plt.show()
```

![](./images/Scatter%20Plot.png)

4. 히스토그램 (Histogram)

사용 상황:
- 데이터의 분포를 나타낼 때
- 빈도수를 확인할 때
- 연속형 데이터의 분포를 분석할 때

```python
import numpy as np

data = np.random.randn(1000)

plt.hist(data, bins=30)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram Example')
plt.show()
```

![](./images/Histogram%20Plot.png)



5. 파이 차트 (Pie Chart)

사용 상황:
- 전체에 대한 부분의 비율을 나타낼 때
- 카테고리별 구성 비율을 시각화할 때

```python
labels = ['A', 'B', 'C', 'D']
sizes = [15, 30, 45, 10]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart Example')
plt.show()
```

![](./images/Pie%20Plot.png)


6. 상자 그림 (Box Plot)

사용 상황:
- 데이터의 분포와 중심 경향, 분산을 나타낼 때
- 이상치(outlier)를 시각화할 때
- 여러 그룹 간의 비교를 할 때

```python
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

plt.boxplot(data, vert=True, patch_artist=True)
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Box Plot Example')
plt.show()
```
![](./images/Box%20Plot.png)


7. 히트맵 (Heatmap)

사용 상황:
- 매트릭스 형태의 데이터를 시각화할 때
- 두 변수 간의 상관관계를 나타낼 때
- 데이터의 패턴을 탐색할 때


```python
import seaborn as sns
import numpy as np

data = np.random.rand(10, 12)
ax = sns.heatmap(data, cmap='YlGnBu')

plt.title('Heatmap Example')
plt.show()
``` 

![](./images/Heatmap%20Plot.png)

8. 바이올린 플롯 (Violin Plot)

사용 상황:
- 데이터의 분포와 밀도를 동시에 나타낼 때
- 상자 그림보다 분포의 모양을 더 잘 보여줄 때

```python
import seaborn as sns
import numpy as np

data = [np.random.normal(0, std, 100) for std in range(1, 4)]
sns.violinplot(data=data)
plt.title('Violin Plot Example')
plt.show()
```

![](./images/Violin%20plot.png)