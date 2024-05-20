# Recommendation System 

## 추천 시스템의 발전 

추천 시스템은 콘텐츠 기반 필터링에서 협업 필터링으로, 그리고 컨텍스트 기반 추천으로 발전해왔습니다

각 단계에서 추천 시스템의 정확성과 사용자 경험을 향상시키기 위한 새로운 기술과 방법론이 도입되었습니다. 아래는 각 방법의 특징과 발전 과정을 간략하게 설명한 것입니다.

콘텐츠 기반 필터링 (Content-Based Filtering): 
- 특징: 사용자에게 아이템(영화, 책, 상품 등)의 속성을 분석하여 유사한 아이템을 추천합니다. 예를 들어, 사용자가 특정 장르의 영화를 좋아하면 유사한 장르의 다른 영화를 추천합니다.
- 장점: 초기 사용자 문제(콜드 스타트 문제)가 적습니다. 사용자의 명시적인 선호도를 기반으로 추천하기 때문에 개인화된 추천이 가능합니다.
- 단점: 새로운 사용자나 아이템에 대해 충분한 데이터가 없는 경우 추천의 품질이 낮아질 수 있습니다. 또한, 사용자가 선호하지 않는 아이템을 추천할 가능성이 있습니다.


협업 필터링 (Collaborative Filtering)
- 특징: 여러 사용자의 과거 행동(예: 영화 평점, 구매 기록 등)을 기반으로 유사한 사용자 그룹을 찾아 추천합니다. 사용자 기반 협업 필터링(user-based)과 아이템 기반 협업 필터링(item-based) 두 가지로 나눌 수 있습니다.
- 장점: 
  - 콘텐츠의 속성에 대한 정보가 없어도 추천이 가능합니다. 사용자 간의 유사성을 기반으로 추천하기 때문에 더 넓은 범위의 추천이 가능합니다.
  - 사용자 또는 아이템간의 유사성을 바탕으로 추천을 생성하는 방법으로 전통적인 머신러닝 모델과는 다름. 그래서 모델 훈련이 필요없음. (물론 ML 모델을 이용해서 추천하는 시스템도 있다.) 
- 단점: 초기 사용자 문제(콜드 스타트 문제)와 희소성 문제(sparsity problem)가 있습니다. 또한, 대규모 데이터를 처리하는 데 시간과 자원이 많이 소요될 수 있습니다.

컨텍스트 기반 추천 (Context-Aware Recommendation): 
- 특징: 사용자의 현재 상황이나 문맥(위치, 시간, 날씨, 기기 등)을 고려하여 추천합니다. 예를 들어, 사용자가 현재 위치한 장소나 시간대에 맞는 추천을 제공합니다.
- 장점: 더 개인화된 추천이 가능하며, 사용자의 현재 필요와 상황에 맞는 추천을 할 수 있습니다. 이는 사용자 경험을 크게 향상시킬 수 있습니다.
- 단점: 컨텍스트 정보를 수집하고 처리하는 데 추가적인 비용과 노력이 필요합니다. 또한, 개인정보 보호 문제를 고려해야 합니다.


#### Q) 사용자 기반 협업 필터링과 아이템 기반 협업 필터링은 뭔데?

사용자 기반 협업 필터링 (User-Based Collaborative Filtering)
- 원리: 
  - 이 방법은 특정 사용자와 유사한 취향을 가진 다른 사용자들을 찾아서 추천을 생성합니다.
  - 예를 들어, 사용자 A가 좋아하는 영화들을 사용자 B도 좋아한다면, 사용자 B가 좋아하는 다른 영화들을 사용자 A에게 추천하는 방식입니다.

- 단계:
  - 유사한 사용자 찾기: 사용자 간의 유사도를 계산합니다. 유사도를 계산하는 데 코사인 유사도(cosine similarity), 피어슨 상관 계수(Pearson correlation coefficient) 등이 사용됩니다.
  - 추천 아이템 선정: 유사한 사용자들이 좋아하는 아이템 중에서 현재 사용자에게 추천할 아이템을 선택합니다.

- 장점:
  - 개인화된 추천이 가능하며, 사용자의 명시적인 선호도를 반영할 수 있습니다.

- 단점:
  - 희소성 문제(sparsity problem): 사용자-아이템 매트릭스가 매우 희소할 경우 유사한 사용자를 찾기 어려울 수 있습니다.
  - 확장성 문제(scalability problem): 사용자가 많아질수록 계산 비용이 증가합니다.


아이템 기반 협업 필터링 (Item-Based Collaborative Filtering)
- 원리:
  - 이 방법은 특정 아이템과 유사한 다른 아이템들을 찾아서 추천을 생성합니다.
  - 예를 들어, 사용자가 A라는 영화를 좋아한다면, A와 유사한 다른 영화들을 추천하는 방식입니다.

- 단계:
  - 유사한 아이템 찾기: 아이템 간의 유사도를 계산합니다. 이 때도 코사인 유사도, 피어슨 상관 계수 등이 사용됩니다.
  - 추천 아이템 선정: 사용자가 좋아하는 아이템들과 유사한 아이템을 추천합니다.

- 장점:
  - 사용자 수에 비해 아이템 수가 상대적으로 적은 경우에 더 효율적입니다.
  - 희소성 문제를 일부 해결할 수 있습니다.

- 단점:
  - 아이템 간의 유사도를 계산하는 데 초기 설정 비용이 발생합니다.
  - 새로운 아이템에 대한 추천이 어려울 수 있습니다(콜드 스타트 문제).


- 확장성 문제, 희소성 문제, 콜드 스타트 문제 등을 고려해야함. 

#### Q) 컨택스트 기반 추천은 사용자의 현재 상황 그리니까 영화 추천으로 들자면 보통 이 사용자는 로맨스를 좋아하는데 오늘따라 공포 영화를 좋아하는 것 같으면 그것에 맞게 빠르게 적응하는 걸 말하는건가?

맞습니다. 

컨텍스트 기반 추천(Context-Aware Recommendation)은 사용자의 현재 상황이나 문맥을 고려하여 추천을 제공하는 방식입니다. 영화 추천의 예를 들어 설명해 보겠습니다

컨텍스트 기반 추천의 원리와 예시
- 원리:
  - 컨텍스트 기반 추천 시스템은 사용자의 현재 상태, 위치, 시간, 날씨, 기기 사용 패턴 등 다양한 상황 정보를 활용합니다.
  - 이러한 정보를 바탕으로 사용자의 현재 필요와 기분에 맞는 추천을 제공합니다.

- 예시:
  - 시간: 사용자가 평일 저녁에는 코미디 영화를 주로 보지만, 주말 밤에는 공포 영화를 즐기는 경향이 있다고 가정해 봅시다. 컨텍스트 기반 추천 시스템은 현재가 주말 밤임을 인식하고 공포 영화를 추천할 수 있습니다.
  - 위치: 사용자가 집에서는 주로 긴 영화를 보고, 출퇴근 시간에는 짧은 TV 시리즈를 선호한다면, 사용자의 현재 위치가 집인지 출퇴근 중인지를 판단하여 적절한 콘텐츠를 추천할 수 있습니다.
  - 날씨: 비오는 날에는 로맨틱 코미디를 선호하고, 맑은 날에는 액션 영화를 선호하는 사용자라면, 현재 날씨를 반영하여 적절한 영화를 추천할 수 있습니다.
  - 기분: 사용자의 기분을 추정할 수 있는 데이터를 활용하여, 사용자가 오늘따라 평소와 다른 장르를 선호할 것 같으면 그에 맞는 영화를 추천할 수 있습니다. 예를 들어, SNS나 최근에 본 영화의 패턴 등을 분석하여 오늘따라 공포 영화를 선호할 것이라고 판단하면 공포 영화를 추천합니다.

- 장점과 단점
  - 장점:
    - 개인화된 경험: 사용자의 현재 상황과 필요에 맞춘 추천을 제공하여 더욱 개인화된 경험을 제공합니다.
    - **적응성: 사용자의 상황 변화에 빠르게 적응하여 추천의 정확성을 높일 수 있습니다.**
  - 단점:
    - 복잡성: 다양한 컨텍스트 정보를 수집하고 처리하는 과정이 복잡하고 비용이 많이 들 수 있습니다. 
    - 개인정보 보호: 사용자의 위치, 시간, 기분 등 민감한 정보를 다루기 때문에 개인정보 보호와 관련된 문제가 발생할 수 있습니다.

    
#### Q) 컨택스트 정보 만으로 사용자의 클릭을 이끌어낼만큼 정교한 추첞이 되기는 하는건가? 협업 필터링 방식이 더 정확도가 높아보여서. 

그래서 여러 시스템을 섞어서 쓰는 하이브리드로 많이 쓴다. 

컨택스트 기반 시스템의 장점은 외부 요인 (위치, 시간, 기분 등)을 기반으로 추천을 해줄 수 있다는 점과, 적응성이 뛰어나서 실시간으로 원하는 것을 추천해줄 수 있다는 점임. 

협업 필터링은 유사성을 바탕으로 좋아할만한 아이템을 추천해줄 수 있다는 것이고. 

이 둘을 섞어서 쓴다면 더 나은 결과를 낼 것. 


#### Q) 머신러닝이나 딥러닝을 이용해서 추천을 할 때는 어떤 모델을 사용해? 

대표적인 예로 행렬 분해(Matrix Factorization)와 신경망 기반 협업 필터링이 있습니다.

행렬 분해(Matrix Factorization): 사용자-아이템 상호작용 매트릭스를 저차원 공간으로 분해하여 유사성을 학습합니다. 대표적인 방법으로는 SVD(Singular Value Decomposition)와 ALS(Alternating Least Squares)가 있습니다.

신경망 기반 협업 필터링: 딥러닝 모델을 사용하여 사용자와 아이템의 잠재 요인(latent factors)을 학습합니다. 예를 들어, 신경망을 통해 사용자와 아이템의 임베딩 벡터를 학습하고, 이를 기반으로 추천을 생성합니다.

#### Q) 협업 필터링에서 유사도 계산 방법 알고리즘은 어떻게 동작해? (e.g 코사인 유사도(Cosine Similarity)) 

코사인 유사도(Cosine Similarity)는 협업 필터링에서 사용자 간 또는 아이템 간의 유사도를 계산하는 데 널리 사용되는 방법입니다. 

두 벡터 간의 유사도를 측정하는 방법으로, 벡터 사이의 코사인 각도를 사용합니다.

코사인 유사도는 0과 1 사이의 값을 가지며, 값이 클수록 두 벡터가 더 유사하다는 것을 의미합니다.

코사인 유사도의 개념과 계산 방법
- 개념: 
  - 두 벡터 간의 코사인 각도를 사용하여 유사도를 측정합니다.
  - 유사도 값은 0과 1 사이에 위치하며, 1에 가까울수록 두 벡터가 매우 유사하다는 것을 의미합니다.
  - 수식적으로는 두 벡터 A와 B의 내적을 그 벡터의 크기의 곱으로 나눈 값입니다.

![](./images/코사인%20유사도.png)

예제: 
- 사용자 간 유사도를 계산하는 예제를 들어보겠습니다. 두 명의 사용자 A와 B가 있으며, 이들이 평가한 영화 평점 데이터가 주어졌다고 가정합니다.

![](./images/코사인%20유사도%202.png)

## 협업 필터링 (CF) 예제: 

영화 리뷰 테이블 구성
- 사용자가 각 영화를 시청하고 리뷰를 남긴 테이블을 Pandas DataFrame으로 구성.


```python
import pandas as pd
import numpy as np
import hashlib
from tqdm import tqdm
from datasets import load_dataset
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 예제 데이터를 생성하는 부분입니다. 
# data 딕셔너리는 사용자들이 영화에 매긴 평점을 나타냅니다. None은 해당 사용자가 그 영화를 평가하지 않았음을 의미합니다.
data = {
    "영화 1": [4, 5, None, 2, 4],
    "영화 2": [None, 4, 3, 3, 4],
    "영화 3": [5, None, 4, 5, 4],
    "영화 4": [3, 3, 5, None, 4],
    "영화 5": [4, 3, 4, 2, None]
}

# 이 부분은 pandas를 사용하여 data 딕셔너리를 데이터프레임으로 변환하는 부분입니다. index 파라미터를 사용하여 각 사용자에게 인덱스를 할당합니다.
# df는 데이터프레임으로, 각 열은 영화, 각 행은 사용자를 나타냅니다.
df = pd.DataFrame(data, index=["사용자 1", "사용자 2", "사용자 3", "사용자 4", "사용자 5"])
```


사용자 유사도 테이블 구성
- Scikit Learn의 코사인 유사도(Cosine Similarity) 함수를 통해, 사용자 유사도 매트릭스 구성.


```python
# df 데이터프레임을 user_df로 복사합니다. 원본 데이터를 보존하고, 복사한 데이터프레임에서 변경 작업을 수행합니다.
user_df = df.copy()

# user_df 데이터프레임에서 NaN 값을 0으로 대체합니다. 이는 평점을 매기지 않은 영화를 0으로 간주하여 유사도 계산을 용이하게 합니다.
user_df = user_df.fillna(0)

# 사용자 간의 코사인 유사도를 계산합니다. user_df의 각 행은 한 명의 사용자를 나타내며, 각 열은 해당 사용자가 매긴 영화 평점을 나타냅니다.
# cosine_similarity(user_df, user_df)는 사용자 간의 유사도를 계산하여 2차원 배열을 반환합니다. 이 배열의 (i, j) 요소는 사용자 i와 사용자 j 간의 유사도를 나타냅니다.
user_similarity = cosine_similarity(user_df, user_df)

# user_similarity 배열을 데이터프레임으로 변환합니다. 인덱스와 열 이름은 원래 user_df의 인덱스를 사용합니다.
# 이렇게 함으로써, 각 사용자 간의 유사도를 쉽게 조회할 수 있는 데이터프레임을 만듭니다.
user_similarity_df = pd.DataFrame(user_similarity, index=user_df.index, columns=user_df.index)
user_similarity_df

# df 데이터프레임을 복사하여 item_df로 저장한 후, 이를 전치(transpose)합니다. 
# 이렇게 하면 행과 열이 바뀌어 각 행이 영화를, 각 열이 사용자를 나타내게 됩니다.
item_df = np.transpose(df.copy())

# item_df 데이터프레임에서 NaN 값을 0으로 대체합니다. 이는 평점을 매기지 않은 사용자의 평가를 0으로 간주하여 유사도 계산을 용이하게 합니다.
item_df = item_df.fillna(0)

# 영화 간의 코사인 유사도를 계산합니다. item_df의 각 행은 한 영화를 나타내며, 각 열은 해당 영화를 평가한 사용자들의 평점을 나타냅니다.
# cosine_similarity(item_df, item_df)는 영화 간의 유사도를 계산하여 2차원 배열을 반환합니다. 이 배열의 (i, j) 요소는 영화 i와 영화 j 간의 유사도를 나타냅니다.
item_similarity = cosine_similarity(item_df, item_df)

# item_similarity 배열을 데이터프레임으로 변환합니다. 인덱스와 열 이름은 원래 item_df의 인덱스를 사용합니다.
item_similarity_df = pd.DataFrame(item_similarity, index=item_df.index, columns=item_df.index)
item_similarity_df
```

사용자 기반 Collaborative Filtering
- 각 사용자의 유사도 정보를 반영하여 결측치의 평점 계산.

```python
# 원본 데이터프레임 df를 full_df로 복사합니다. 이 복사본을 수정하여 누락된 평점을 채울 것입니다.
full_df = df.copy()

# full_df 데이터프레임의 각 사용자(행)를 순회합니다. user_id는 현재 순회 중인 사용자의 인덱스입니다.
for user_id in full_df.index:
    # 현재 사용자의 각 영화(열)를 순회합니다. movie_id는 현재 순회 중인 영화의 열 이름입니다.
    for movie_id in full_df.columns:
        
        # 현재 사용자가 해당 영화를 이미 평가했다면, 다음 영화로 넘어갑니다. 즉, 현재 영화 평점이 NaN이 아니면 건너뜁니다.
        if not np.isnan(full_df[movie_id][user_id]): continue

        # user_id 사용자와 다른 사용자들 간의 유사도 값을 가져옵니다. user_similarity_df에서 현재 사용자의 유사도를 복사합니다.
        similarities = user_similarity_df[user_id].copy()
        
        # 현재 영화에 대한 모든 사용자의 평점을 가져옵니다.
        movie_ratings = full_df[movie_id].copy()

        # movie_ratings에서 NaN 값이 있는 인덱스를 추출합니다. 이는 해당 영화를 평가하지 않은 사용자들의 인덱스입니다.
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        
        # movie_ratings에서 NaN 값을 제거하여, 평점이 있는 값들만 남깁니다.
        movie_ratings = movie_ratings.dropna()
        
        # similarities에서 NaN 평점을 가진 사용자들의 유사도를 제거합니다. 이는 평점이 있는 사용자들만 남기기 위함입니다.
        similarities = similarities.drop(none_rating_idx)

        # 현재 영화의 누락된 평점을 예측합니다. similarities와 movie_ratings 간의 점곱(dot product)을 계산하고, 이를 유사도의 합으로 나누어 가중평균을 구합니다.
        # np.dot(similarities, movie_ratings)는 유사도를 평점에 곱하여 합산합니다.
        # similarities.sum()은 유사도의 합입니다.
        # 이를 나누어 가중평균 평점을 계산합니다.
        # np.dot(similarities, movie_ratings) / similarities.sum() 계산은 그러니까 나와 유사도가 높은 사용자의 평점은 신뢰해서 매겨지고, 나와 유사도가 낮은 사용자의 평점은 낮게 측정되서 평균이 매겨진다.  
        mean_rating = np.dot(similarities, movie_ratings) / similarities.sum()
        
        # 예측한 평점을 full_df의 해당 위치에 채워 넣습니다. mean_rating을 반올림하여 정수로 저장합니다.  
        full_df[movie_id][user_id] = round(mean_rating)

full_df
```

```python
# 사용자가 이미 본 영화 목록을 정의합니다.
# watched_movies 리스트에는 사용자가 이미 본 두 개의 영화 "영화 4"와 "영화 3"이 포함되어 있습니다.
watched_movies = ["영화 4", "영화 3"]

# item_similarity_df 데이터프레임에서 사용자가 본 첫 번째 영화("영화 4")와 유사한 영화들의 유사도 값을 가져옵니다.
# item_similarity_df["영화 4"]는 "영화 4"와 다른 모든 영화들 간의 유사도 값을 포함하는 시리즈입니다.
# item_similarity_df.index.isin(watched_movies)는 item_similarity_df의 인덱스(영화 목록)가 watched_movies 리스트에 있는지 여부를 나타내는 불리언 시리즈를 반환합니다.
# ~ 연산자는 이 불리언 시리즈를 반전시켜, 사용자가 이미 본 영화들이 아닌 영화들을 선택합니다.
# 위의 반전된 불리언 시리즈를 사용하여, 사용자가 이미 본 영화들이 아닌 영화들에 대한 유사도 값을 선택합니다.
# 이 부분은 "영화 4"와 유사한, 그러나 사용자가 아직 보지 않은 영화들의 유사도 값을 반환합니다.
# .sort_values(ascending=False): 선택된 유사도 값을 내림차순으로 정렬합니다.
# 유사도가 높은 영화부터 낮은 영화 순으로 정렬됩니다.
# 가장 유사도가 높은 상위 3개의 영화를 선택합니다.
# 이를 통해 "영화 4"와 가장 유사한, 사용자가 아직 보지 않은 상위 3개의 영화를 추천합니다.
item_similarity_df[watched_movies[0]][~item_similarity_df.index.isin(watched_movies)].sort_values(ascending=False)[:3]
```

MovieLens 1M 데이터셋을 로드하고, 데이터셋을 청크로 나누어 데이터프레임으로 변환하는 과정을 수행합니다. 

```python
# datasets 라이브러리에서 load_dataset 함수를 사용하여 "nbtpj/movielens-1m-ratings" 데이터셋을 로드합니다.
# 데이터셋의 "train" 부분을 가져오고, shuffle(seed=10)을 사용하여 데이터를 섞습니다.
# seed=10은 셔플링의 결과가 재현 가능하도록 고정된 시드를 사용합니다.
data = load_dataset("nbtpj/movielens-1m-ratings")["train"].shuffle(seed=10)

# 데이터를 나눌 청크(chunks)의 개수를 100으로 설정합니다.
n_chunks = 100

# 데이터셋을 100개의 청크로 나누었을 때, 각 청크의 크기를 계산합니다. len(data) // n_chunks는 데이터셋의 전체 길이를 100으로 나눈 몫입니다.
chunk_size = len(data) // n_chunks

# 데이터셋의 길이가 청크 개수로 나누어 떨어지지 않을 경우, 마지막 청크의 크기를 조정합니다.
# len(data) % n_chunks != 0 조건이 참이면, chunk_size에 1을 더하여 마지막 청크에 남은 데이터를 모두 포함할 수 있도록 합니다.
if len(data) % n_chunks != 0: chunk_size += 1

# 빈 데이터프레임 movielens_df를 생성합니다. 이후 청크 데이터프레임들을 이곳에 병합할 것입니다.
movielens_df = pd.DataFrame()

# tqdm을 사용하여 청크를 생성하는 루프를 진행하며, 진행 상황을 시각적으로 표시합니다.
# range(0, len(data), chunk_size)는 0부터 데이터셋 길이까지 chunk_size 간격으로 범위를 설정합니다. start는 각 청크의 시작 인덱스입니다.
for start in tqdm(range(0, len(data), chunk_size)):
    # 현재 청크의 끝 인덱스를 계산합니다. start에 chunk_size를 더하여 끝 인덱스를 구합니다.
    end = start + chunk_size
    
    # 데이터셋의 현재 청크 범위를 슬라이싱하여 데이터프레임 chunk_df로 변환합니다.
    # data[start:end]는 현재 청크에 해당하는 데이터를 슬라이싱한 결과입니다.
    chunk_df = pd.DataFrame(data[start:end])
    
    # movielens_df와 chunk_df를 병합합니다.
    # pd.concat 함수는 두 데이터프레임을 연결하며, ignore_index=True는 인덱스를 무시하고 새로운 연속 인덱스를 생성합니다.
    movielens_df = pd.concat([movielens_df, chunk_df], ignore_index=True)

# movielens_df 데이터프레임에서 필요한 열만 선택합니다.
# ["user_id", "movie_id", "user_rating"] 열만 포함하여 새로운 데이터프레임을 생성합니다.
movielens_df = movielens_df[["user_id", "movie_id", "user_rating"]]
```

movielens_df 데이터프레임을 사용하여 사용자-아이템 매트릭스를 생성하는 과정입니다

```python
# 데이터프레임을 피벗 테이블로 변환:
# movielens_df는 사용자, 영화, 평점 데이터를 포함하는 데이터프레임입니다.
# pivot_table 메소드를 사용하여 데이터를 피벗 테이블 형태로 변환합니다.
# index="user_id": 사용자 ID를 행(index)로 설정합니다.
# columns="movie_id": 영화 ID를 열(columns)로 설정합니다.
# values="user_rating": 각 셀의 값으로 사용자 평점을 설정합니다.
user_item_matrix = movielens_df.pivot_table(index="user_id", columns="movie_id", values="user_rating")
user_item_matrix
```


#### Q) 100 개의 청크로 나눠서 각 청크를 데이터 프레임으로 변환해서 하나의 큰 데이터프레임을 구성하는 이유는 뭐야? 청크로 안나누고 그냥 만들면 안돼?

데이터셋이 작아서 메모리에 무리가 가지 않는 경우라면, 청크로 나누지 않고 한 번에 로드하여 처리할 수도 있습니다. 이 경우 코드가 더 간단해집니다.
```python
data = load_dataset("nbtpj/movielens-1m-ratings")["train"].shuffle(seed=10)
movielens_df = pd.DataFrame(data)
movielens_df = movielens_df[["user_id", "movie_id", "user_rating"]]
```

#### Q) 피벗 테이블이란 뭔가? 

피벗 테이블 변환은 원본 데이터프레임의 행과 열을 재배치하여 집계된 데이터를 생성하는 과정입니다. 

이를 통해 데이터를 더 쉽게 분석하고 다양한 관점에서 볼 수 있습니다.

예시로 보면 쉽다. 

예제 설명
- 예제에서는 사용자-아이템 매트릭스를 만들기 위해 피벗 테이블을 사용합니다.
- 원본 데이터 (movielens_df):

```text
   user_id  movie_id  user_rating
0        1         1            4
1        1         3            5
2        1         4            3
3        1         5            4
4        2         1            5
5        2         2            4
6        2         4            3
7        2         5            3
8        3         2            3
9        3         3            4
10       3         4            5
11       3         5            4
12       4         1            2
13       4         2            3
14       4         3            5
15       4         5            2
16       5         1            4
17       5         2            4
18       5         3            4
19       5         4            4
```

이 데이터를 피벗 테이블로 변환하여 사용자-아이템 매트릭스를 만듭니다.

```python
user_item_matrix = movielens_df.pivot_table(index="user_id", columns="movie_id", values="user_rating")
```

```text
movie_id      1    2    3    4    5
user_id
1           4.0  NaN  5.0  3.0  4.0
2           5.0  4.0  NaN  3.0  3.0
3           NaN  3.0  4.0  5.0  4.0
4           2.0  3.0  5.0  NaN  2.0
5           4.0  4.0  4.0  4.0  NaN
```

사용자 기반 협업 필터링을 사용하여 사용자 간의 유사도를 계산하고, 특정 사용자와 영화에 대한 평점을 예측하는 함수 predict_rating을 정의합니다

```python
# user_item_matrix의 결측값(NaN)을 0으로 채웁니다.
# csr_matrix는 희소 행렬(Sparse Matrix) 형식으로 데이터를 변환합니다. 이는 메모리 효율성을 높이기 위해 사용됩니다.
# cosine_similarity 함수를 사용하여 사용자 간의 코사인 유사도를 계산합니다. 이 함수는 두 벡터 간의 코사인 각도를 계산하여 유사도를 측정합니다.
# 결과적으로 user_similarity는 사용자 간의 유사도를 나타내는 행렬이 됩니다.
user_similarity = cosine_similarity(csr_matrix(user_item_matrix.fillna(0)))

# user_similarity 배열을 데이터프레임으로 변환합니다.
# 데이터프레임의 인덱스와 열 이름을 user_item_matrix의 인덱스로 설정하여, 각 사용자의 유사도를 쉽게 조회할 수 있도록 합니다.
user_similarity = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# 특정 사용자(user_id)가 특정 영화(movie_id)에 대해 예상 평점을 계산하는 함수 predict_rating을 정의합니다.
def predict_rating(user_id: str, movie_id: str):
    
    # movie_ratings는 user_item_matrix에서 해당 movie_id 열의 데이터를 가져옵니다. 이는 각 사용자가 해당 영화에 대해 매긴 평점입니다.
    movie_ratings = user_item_matrix[movie_id]
    
    # user_similarities 는 user_similarity에서 해당 user_id 행의 데이터를 가져옵니다. 이는 다른 사용자들과의 유사도를 나타냅니다.
    user_similarities = user_similarity[user_id]
    
    # valid_indices는 movie_ratings에서 결측값이 아닌 인덱스를 나타내는 불리언 시리즈입니다. 이는 해당 영화를 실제로 평가한 사용자들을 나타냅니다.
    valid_indices = movie_ratings.notna()
    
    # movie_ratings를 결측값이 아닌 값들만 남겨두고 필터링합니다.
    movie_ratings = movie_ratings[valid_indices]
    
    # user_similarities를 결측값이 아닌 인덱스와 동일하게 필터링합니다. 이는 해당 영화를 평가한 사용자들과의 유사도만 남겨두는 것입니다.
    user_similarities = user_similarities[valid_indices]
    
    # user_similarities의 합이 0보다 큰지 확인합니다. 이는 유사도가 0보다 큰 사용자가 한 명 이상 있는지 확인하는 것입니다.
    if user_similarities.sum() > 0:
        # 유사도 가중 평균을 계산하여 예상 평점을 반환합니다.
        # np.dot(user_similarities, movie_ratings)는 유사도와 평점을 곱한 값들의 합을 계산합니다.
        # user_similarities.sum()은 유사도의 합입니다.
        # 이 둘을 나누어 가중 평균을 구합니다.
        return np.dot(user_similarities, movie_ratings) / user_similarities.sum()
    
    # 유사도가 0보다 큰 사용자가 없으면, 해당 영화의 전체 평균 평점을 반환합니다.
    return movielens_df[movielens_df["movie_id"] == movie_id]["user_rating"].mean()

user_similarity
```



#### Q) csr_matrix 함수는 뭐야?

희소 행렬(Sparse Matrix)을 압축된 형태로 표현하는 기능을 제공합니다. 

메모리를 효율적으로 사용하기 위해 이 함수를 사용한다.

희소 행렬은 대부분의 요소가 0인 행렬입니다

이러한 행렬을 일반적인 밀집 행렬(Dense Matrix) 형식으로 저장하면 많은 메모리가 낭비됩니다. 대신, 희소 행렬 형식을 사용하면 메모리 사용량을 크게 줄일 수 있습니다.

다음은 csr_matrix를 사용하는 간단한 예제입니다:

```python
import numpy as np
from scipy.sparse import csr_matrix

# 밀집 행렬(Dense Matrix)
dense_matrix = np.array([
    [0, 0, 3, 0],
    [2, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1, 0, 4]
])

# 희소 행렬로 변환
sparse_matrix = csr_matrix(dense_matrix)

print(sparse_matrix)
```

출력: 

```text
  (0, 2)    3
  (1, 0)    2
  (3, 1)    1
  (3, 3)    4
```

전체 데이터셋을 학습 데이터와 테스트 데이터로 분리하고, 테스트 데이터를 사용하여 협업 필터링 모델의 예측 성능을 평가합니다.

특히, 예측된 평점과 실제 평점 간의 정확도를 계산하고 누적 정확도를 시각화합니다. 


```python
# movielens_df 데이터프레임을 학습 데이터와 테스트 데이터로 분리합니다.
# train_test_split 함수는 데이터를 랜덤하게 분할합니다.
# test_size=0.02는 전체 데이터의 2%를 테스트 데이터로 사용함을 의미합니다.
# random_state=10은 랜덤 시드를 고정하여 재현 가능한 결과를 보장합니다.
train_data, test_data = train_test_split(movielens_df, test_size=0.02, random_state=10)

# 예측 평점을 저장할 빈 리스트 predictions와 실제 평점을 저장할 빈 리스트 true_ratings를 생성합니다.
# 각 리스트는 float 타입의 요소를 가집니다.
predictions: list[float] = []
true_ratings: list[float] = []

# tqdm을 사용하여 진행 상황을 시각적으로 표시하면서 테스트 데이터의 각 행을 반복합니다.
# test_data.iterrows()는 테스트 데이터프레임의 각 행을 반복할 수 있게 합니다.
# total=test_data.shape[0]는 진행 상황 표시를 위한 총 행 수를 지정합니다.
for idx, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
    # predict_rating 함수를 호출하여 현재 행(row)의 사용자(user_id)와 영화(movie_id)에 대한 예측 평점을 계산합니다.
    prediction = predict_rating(row["user_id"], row["movie_id"])
    
    # 예측된 평점을 predictions 리스트에 추가합니다.
    predictions.append(prediction)
    
    # 실제 평점을 true_ratings 리스트에 추가합니다.
    true_ratings.append(row["user_rating"])

# predictions 리스트의 모든 평점을 반올림하여 정수로 변환하고, rounded_predictions 리스트에 저장합니다.
rounded_predictions = [round(pred) for pred in predictions]

# accuracy_score 함수를 사용하여 실제 평점(true_ratings)과 반올림된 예측 평점(rounded_predictions) 간의 정확도를 계산합니다.
# 실제 평점과 예측 평점이 아예 동일하면 100% 가 나올것임. 
# 예측 값이 실제 값과 정확히 일치하는 경우에만 맞는 것으로 간주합니다. 
accuracy = accuracy_score(true_ratings, rounded_predictions)

# 계산된 정확도를 출력합니다.
print(f"Accuracy: {accuracy}")

# 누적 정확도를 계산합니다.
# np.array(rounded_predictions) == np.array(true_ratings)는 각 예측이 실제 값과 일치하는지를 나타내는 불리언 배열을 생성합니다.
# np.cumsum은 이 불리언 배열의 누적 합을 계산합니다.
# (np.arange(len(test_data)) + 1)는 각 테스트 인스턴스의 인덱스 배열을 생성하여, 누적 합을 해당 인덱스와 나누어 누적 정확도를 계산합니다.
cumulative_accuracy = np.cumsum(np.array(rounded_predictions) == np.array(true_ratings)) / (np.arange(len(test_data)) + 1)

plt.figure(figsize=(10, 5))
plt.plot([i for i in range(len(test_data))], cumulative_accuracy, linewidth=2, color="#fc1c49")
plt.title("Cumulative Accuracy of predictions")
plt.xlabel("Test Instance")
plt.ylabel("Cumulative Accuracy")
plt.show()
```

