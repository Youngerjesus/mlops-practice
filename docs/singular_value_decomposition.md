# Singular Value Decomposition 

Singular Value Decomposition (SVD)는 행렬을 세 개의 다른 행렬의 곱으로 분해하는 중요한 선형 대수 기법입니다

이는 데이터 압축, 노이즈 감소, 추천 시스템 등 여러 분야에서 널리 사용됩니다.

![](./images/SVD%201.png)
![](./images/SVD%202.png)

SVD의 주요 응용: 
- 차원 축소(Dimensionality Reduction):
  - 데이터의 차원을 줄여 계산 비용을 낮추고, 노이즈를 감소시키는 데 사용됩니다.
  - Principal Component Analysis (PCA)에서 SVD는 데이터를 주성분으로 변환하는 데 사용됩니다.

- 데이터 압축(Data Compression):
  - 데이터의 중요한 부분을 유지하면서 원래 데이터의 크기를 줄이는 데 사용됩니다.

- 노이즈 감소(Noisy Reduction):
  - 데이터에서 노이즈를 제거하여 중요한 패턴을 찾는 데 유용합니다.

- 추천 시스템(Recommendation Systems):
  - 사용자-아이템 행렬을 분해하여 잠재 요인을 추출하고, 이를 통해 사용자에게 적절한 아이템을 추천합니다.

### 예시 코드 

데이터 준비 코드 

```python

import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

data = load_dataset("nbtpj/movielens-1m-ratings")["train"].shuffle(seed=10).select(range(200000))
movielens_df = pd.DataFrame(data)
movielens_df = movielens_df[["user_id", "movie_id", "user_rating"]]

user_ids = movielens_df["user_id"].unique()
user_id_map = {id: index for index, id in enumerate(user_ids)}
movie_ids = movielens_df["movie_id"].unique()
movie_id_map = {id: index for index, id in enumerate(movie_ids)}

movielens_df["user_id"] = movielens_df["user_id"].map(user_id_map)
movielens_df["movie_id"] = movielens_df["movie_id"].map(movie_id_map)
```

영화 추천 시스템을 구축하기 위해 사용자-영화 평점 행렬을 생성하고, 이를 SVD를 사용하여 분해하는 과정 코드

```python
# train_test_split 함수를 사용하여 movielens_df 데이터를 학습 데이터(train_data)와 테스트 데이터(test_data)로 나눕니다. 
# 테스트 데이터의 비율은 20%이며, random_state를 10으로 설정하여 결과의 재현성을 보장합니다.
train_data, test_data = train_test_split(movielens_df, test_size=0.2, random_state=10)

# train_data에서 각 사용자(user_id)별로 평균 평점(user_rating)을 계산하여 딕셔너리 형태로 저장합니다. 이는 사용자가 평가하지 않은 영화의 평점을 대체하기 위해 사용됩니다.
user_avg_ratings = train_data.groupby('user_id')['user_rating'].mean().to_dict()

# train_data를 피벗 테이블로 변환하여 사용자-영화 평점 행렬을 생성합니다. 
# user_id를 행으로, movie_id를 열로 설정하고, 각 셀에는 해당 사용자가 해당 영화에 부여한 평점(user_rating)이 들어갑니다. 
# 평점이 없는 셀은 해당 사용자의 평균 평점으로 채웁니다.
ratings_matrix = train_data.pivot(index="user_id", columns="movie_id", values="user_rating").apply(lambda x: x.fillna(user_avg_ratings[x.name]), axis=1)

# 각 사용자의 평균 평점을 계산하여 user_rating_mean에 저장합니다. 이는 행(사용자) 기준으로 계산됩니다. 
user_rating_mean = ratings_matrix.mean(axis=1)

# 각 사용자의 평점에서 해당 사용자의 평균 평점을 뺀 값을 ratings_matrix_demeaned에 저장합니다. 이는 사용자 간의 평가 편향을 제거하기 위한 과정입니다.
ratings_matrix_demeaned = ratings_matrix - user_rating_mean.values.reshape(-1, 1)

# ratings_matrix_demeaned를 csr_matrix 형식의 희소 행렬로 변환합니다. 이는 효율적인 행렬 연산을 가능하게 합니다.
ratings_matrix_csr = csr_matrix(ratings_matrix_demeaned.values)

# svds 함수를 사용하여 ratings_matrix_csr의 특이값 분해(SVD)를 수행합니다. k=200은 200개의 특이값 및 관련 벡터를 계산하라는 의미입니다. 결과는 U, sigma, Vt로 분해됩니다.
U, sigma, Vt = svds(ratings_matrix_csr, k=200)

# sigma 벡터를 대각 행렬로 변환합니다. 이 대각 행렬은 특이값을 나타내며, SVD에서의 축소된 차원 공간을 형성합니다.
sigma = np.diag(sigma)
```

SVD를 사용하여 영화 평점을 예측하고, 테스트 데이터에 대해 예측된 평점과 실제 평점의 차이를 계산하여 모델의 성능을 평가하는 코드 

```python
# U, sigma, Vt 행렬을 사용하여 전체 사용자에 대한 예측 평점 행렬을 계산합니다. 
# 이 때, U와 sigma, Vt의 행렬 곱을 수행한 후 각 사용자별 평균 평점을 더해줍니다. 
# np.dot는 행렬 곱셈을 수행합니다. user_rating_mean.values.reshape(-1, 1)은 평균 평점을 각 행에 맞게 재배열합니다.
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_rating_mean.values.reshape(-1, 1)

# 모든 예측 평점의 평균을 계산하여 user_mean에 저장합니다. 이는 예측할 수 없는 경우에 사용할 기본 평점을 제공합니다.
user_mean = np.mean(all_user_predicted_ratings)

# 주어진 사용자 ID와 영화 ID에 대해 예측 평점을 반환하는 함수입니다. 만약 주어진 ID가 예측 평점 행렬의 크기를 벗어나는 경우, user_mean을 반환합니다.
def predict_rating_svd(user_id: int, movie_id: int):
    if user_id < all_user_predicted_ratings.shape[0] and movie_id < all_user_predicted_ratings.shape[1]:
        return all_user_predicted_ratings[user_id, movie_id]
    else:
        return user_mean

# 예측 평점과 실제 평점을 저장할 리스트를 각각 초기화합니다.    
predictions: list[float] = []
true_ratings: list[float] = []


# test_data의 각 행을 반복합니다. tqdm을 사용하여 진행 상황을 시각적으로 보여줍니다. total=test_data.shape[0]는 총 반복 횟수를 나타냅니다.
for _, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
    
    # 테스트 데이터의 현재 행에서 user_id, movie_id, user_rating 값을 추출하여 각각 user_id, movie_id, true_rating 변수에 저장합니다.
    user_id = int(row["user_id"])
    movie_id = int(row["movie_id"])
    true_rating = row["user_rating"]

    # predict_rating_svd 함수를 사용하여 해당 사용자와 영화에 대한 예측 평점을 계산합니다.
    predicted_rating = predict_rating_svd(user_id, movie_id)
    
    # 예측 평점과 실제 평점을 각각 predictions와 true_ratings 리스트에 추가합니다. 예측 평점은 반올림하여 저장합니다.
    predictions.append(round(predicted_rating))
    true_ratings.append(true_rating)

# 예측 평점과 실제 평점 간의 루트 평균 제곱 오차(RMSE)를 계산합니다. 이는 모델의 성능을 평가하는 지표로 사용됩니다.
# 계산된 RMSE 값을 출력합니다.
rmse = np.sqrt(np.mean((np.array(predictions) - np.array(true_ratings))**2))
print(f"RMSE: {rmse}")
```


### Q) SVD 를 추천 시스템에 적용할 때는 훈련을 하는 코드 같은건 없는거야?

없다. 

SVD 행렬 분해 자체가 모델 훈련으로 생각하면 됨. 이를 통해 얻은 잠재 요인을 사용하여 평점을 예측하고 성능을 평가합니다

SVD(특이값 분해)를 추천 시스템에 적용할 때 훈련(train)이라고 하는 과정은 명시적인 학습 과정이 아닌, 데이터의 행렬 분해를 통해 잠재 요인을 추출하는 것을 의미합니다. 이는 주어진 사용자-아이템 평점 행렬을 저차원으로 축소하여 잠재 요인을 찾고, 이를 통해 추천을 수행하는 방식입니다.

