# Neural Collaborative Filtering 

Neural Collaborative Filtering (NCF)는 추천 시스템을 구성하는 방법 중 하나로, 전통적인 행렬 분해 기반의 방법과 인공 신경망(Neural Network)을 결합하여 추천 성능을 향상시키는 기법입니다. 

NCF는 사용자와 항목 간의 상호작용 데이터를 학습하여 추천 모델을 구축하는데, 이 과정에서 딥러닝 기법을 활용하여 더 복잡한 비선형 관계를 학습할 수 있습니다.

NCF의 주요 구성 요소는 다음과 같습니다:
- Embedding Layer: 사용자와 아이템을 각각 고정된 크기의 임베딩 벡터로 변환합니다. 이 임베딩 벡터는 사용자나 아이템의 고유한 특징을 학습합니다.
- Hidden Layers: 임베딩 벡터를 입력으로 받아 여러 개의 은닉층을 통과하면서 사용자와 항목 간의 복잡한 상호작용을 학습합니다. 이 과정에서 비선형 활성화 함수가 사용되어 복잡한 패턴을 학습합니다.
- Output Layer: 마지막 은닉층의 출력을 통해 사용자와 항목 간의 상호작용을 예측합니다. 이 예측값은 주로 사용자가 항목을 얼마나 좋아할지를 나타내는 점수로 사용됩니다.

NCF 모델은 일반적으로 다음과 같은 구조로 구현됩니다:
- Input Layer: 사용자와 항목 ID를 임베딩 레이어에 전달합니다.
- Embedding Layer: 사용자와 항목을 임베딩 벡터로 변환합니다.
- Interaction Layer: 사용자 임베딩과 항목 임베딩을 결합하여 상호작용 벡터를 생성합니다.
- Hidden Layers: 여러 개의 은닉층을 통해 상호작용 벡터를 전달하며 학습합니다.
- Output Layer: 최종 예측값을 생성합니다.

예시 코드: 
- 다음은 NCF의 간단한 구현 예시입니다.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

# 임베딩 차원 설정
# 사용자의 임베딩 벡터와 아이템의 임베딩 벡터의 차원을 50으로 설정합니다. 즉, 각 사용자와 아이템은 50차원의 벡터로 표현됩니다.
embedding_dim = 50

# 사용자와 항목의 수 설정
# 총 1000명의 사용자와 1500개의 아이템이 있다고 가정합니다. 이 값들은 임베딩 레이어의 입력 크기를 설정하는 데 사용됩니다.
num_users = 1000
num_items = 1500

# 입력 레이어
# 사용자 ID와 아이템 ID를 입력으로 받는 입력 레이어를 각각 정의합니다. 각 입력은 단일 값(사용자 ID 또는 아이템 ID)을 가집니다.
user_input = Input(shape=(1,), name='user_input')
item_input = Input(shape=(1,), name='item_input')

# 임베딩 레이어
# 사용자와 아이템 ID를 임베딩 벡터로 변환하는 임베딩 레이어를 정의합니다. 각 사용자와 아이템 ID는 50차원의 임베딩 벡터로 변환됩니다.
user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding')(user_input)
item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim, name='item_embedding')(item_input)

# Flatten 임베딩 레이어
# 임베딩 벡터를 평탄화(flatten)하여 2차원 배열에서 1차원 배열로 변환합니다. 이는 후속 레이어에 입력으로 사용될 벡터입니다.
user_vector = Flatten()(user_embedding)
item_vector = Flatten()(item_embedding)

# Concatenate 임베딩 벡터
# 사용자와 아이템의 임베딩 벡터를 결합(concatenate)하여 하나의 벡터로 만듭니다. 이 벡터는 모델의 은닉층으로 입력됩니다.
concat = Concatenate()([user_vector, item_vector])

# 은닉층
# 첫 번째 은닉층은 128개의 뉴런을 가지며, ReLU 활성화 함수를 사용합니다. 결합된 임베딩 벡터를 입력으로 받습니다.
# 두 번째 은닉층은 64개의 뉴런을 가지며, ReLU 활성화 함수를 사용합니다. 첫 번째 은닉층의 출력을 입력으로 받습니다.
dense_1 = Dense(128, activation='relu')(concat)
dense_2 = Dense(64, activation='relu')(dense_1)

# 출력 레이어
# 출력 레이어는 1개의 뉴런을 가지며, Sigmoid 활성화 함수를 사용합니다. 이 레이어는 최종 예측 값을 출력합니다. Sigmoid 함수는 출력값을 0과 1 사이로 제한하므로, 이 모델은 이진 분류나 확률 예측에 적합합니다.
output = Dense(1, activation='sigmoid')(dense_2)

# 모델 생성
# 정의된 입력 레이어와 출력 레이어를 사용하여 모델을 생성합니다. 이 모델은 사용자와 아이템 ID를 입력으로 받아서 예측 값을 출력합니다.
model = Model(inputs=[user_input, item_input], outputs=output)

# 모델 컴파일
# 모델을 컴파일합니다. Adam 옵티마이저를 사용하며 학습률은 0.001로 설정합니다. 손실 함수로는 Mean Squared Error를 사용합니다.
model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

# 더미 데이터 생성 (예시)
num_samples = 10000
user_data = np.random.randint(0, num_users, size=(num_samples, 1))
item_data = np.random.randint(0, num_items, size=(num_samples, 1))
ratings = np.random.rand(num_samples, 1)

# 모델 학습
# 생성된 데이터를 사용하여 모델을 학습시킵니다. 학습은 5 epoch 동안 진행되며, 배치 크기는 32로 설정합니다.
model.fit([user_data, item_data], ratings, epochs=5, batch_size=32)

# 모델 요약 출력
model.summary()
```

학습 과정: 
- 학습 데이터와 레이블
  - 학습 데이터: 각 학습 샘플은 사용자 ID, 아이템 ID, 그리고 해당 사용자와 아이템 간의 실제 평점으로 구성됩니다.
    - 예: (사용자 1, 아이템 2, 평점 4), (사용자 1, 아이템 3, 평점 5)
  - 레이블: 각 샘플의 레이블은 해당 사용자와 아이템 간의 실제 평점입니다.
    - 예: 사용자 1이 아이템 2에 대해 4점을 주었고, 아이템 3에 대해 5점을 주었다면, 이러한 평점들이 레이블로 사용됩니다.

- 학습 과정
  - 1. 데이터 준비
    - 각 샘플은 사용자 ID와 아이템 ID를 입력으로 받습니다.
    - 해당 사용자와 아이템 간의 실제 평점은 레이블로 사용됩니다.
  - 2. 모델 입력
    - 사용자 ID와 아이템 ID가 임베딩 벡터로 변환됩니다.
    - 임베딩 벡터는 모델의 은닉층을 거쳐 최종 예측 값을 생성합니다.

  - 3. 손실 계산
    - 모델의 예측 값과 실제 평점 간의 차이를 손실 함수로 계산합니다.
    - 예를 들어, 모델이 사용자 1이 아이템 2에 대해 4.5점을 예측했지만 실제 평점은 4점인 경우, 손실은 4.5와 4의 차이로 계산됩니다.
  - 4. 모델 업데이트
    - 손실의 기울기를 계산하여 모델의 파라미터(임베딩 벡터 포함)를 업데이트합니다.
    - 이 과정이 반복되면서 모델은 점점 더 정확하게 예측할 수 있게 됩니다.

- 예측
  - 모델이 학습된 후, 새로운 입력 데이터에 대해 예측을 수행할 수 있습니다. 예를 들어, 사용자 1에 대해 학습된 모델이 아이템 1의 평점을 예측할 수 있습니다.

모델 업데이트는 사용자 임베딩 벡터 업데이트, 아이템 임베딩 벡터 업데이트, 모델의 가중치 업데이트 이 셋을 업데이트 된다.
- 기본적으로 학습 과정에서 모든 업데이트는 손실 함수의 기울기에 의해 결정되지만, 특정 상황에서 어떤 업데이트가 더 많이 일어나는 부분이 있을 것이다.
- 서로 극명하게 다른 사용자가 같은 아이템을 다른 평점으로 평가한 데이터의 경우에는 사용자 임베딩 계산 쪽을 업데이트 해줄 거고
- 한 사용자가 극명하게 다른 아이템에 대해 다른 평점으로 평가한 데이터의 경우에는 아이템 임베딩 쪽을 업데이트 해줄 거고
- 전체적으로 평점 예측이 실제 결과에 비해 다운되어있다면 모델 가중치 쪽을 업데이트 해줄거고 




### Q) 사용자와 아이템 모두 임베딩 벡터로 변환하는거면 유사한 사용자나 유사한 아이템은 비슷한 벡터로 변환되는거야?

맞습니다. 

Neural Collaborative Filtering(NCF)에서는 사용자와 아이템을 임베딩 벡터로 변환하는 과정에서 유사한 사용자나 유사한 아이템은 비슷한 벡터로 변환됩니다.

임베딩의 개념:
- 임베딩 벡터는 고차원의 데이터를 저차원의 연속 공간에 매핑하는 것을 의미합니다. 이 과정에서 임베딩 레이어는 학습 데이터에 기반하여 비슷한 특성을 가진 항목을 유사한 벡터 공간에 배치하게 됩니다.

유사성 학습:
- 학습 과정에서 모델은 사용자-아이템 상호작용 데이터를 통해 유사한 사용자들이 유사한 아이템을 좋아한다는 패턴을 학습합니다. 이를 통해 유사한 사용자들은 임베딩 공간에서 비슷한 위치에 배치됩니다.
- 마찬가지로, 유사한 아이템들도 비슷한 위치에 배치되어 유사한 임베딩 벡터를 가지게 됩니다.


### Q) 처음에는 사용자와 아이템의 임베딩 벡터를 임의의 벡터로 설정하고, 학습 과정을 거치면서 사용자 임베딩 벡터로 설정하는 과정과, 아이템을 임베딩 벡터로 올바르게 설정하게 되는거야?

맞습니다. 

Neural Collaborative Filtering (NCF) 모델에서는 처음에 사용자와 아이템의 임베딩 벡터를 임의의 값으로 초기화합니다. 학습 과정에서 이 임베딩 벡터들은 사용자가 아이템과 상호작용한 데이터를 통해 점차적으로 조정됩니다.

이를 통해 사용자 임베딩 벡터와 아이템 임베딩 벡터가 점점 더 올바르게 설정됩니다. 이 과정을 자세히 설명하면 다음과 같습니다

초기화 및 학습 과정
- 1. 임의 초기화:
  - 모델이 처음 시작할 때, 사용자와 아이템의 임베딩 벡터는 무작위 값으로 설정됩니다. 일반적으로 이 값들은 작은 무작위 수로 초기화됩니다.

- 2. 학습 과정:
  - 학습 데이터는 사용자와 아이템 간의 상호작용(예: 사용자 A가 아이템 X를 평가한 평점)을 포함합니다



### Q) NCF 는 그러면 사용자를 구별시켜줄 수 있는 사용자 특성 데이터와 아이템을 구별시켜줄 수 있는 아이템 특성 데이터가 초기에 있다라고 가정하고, 이 특성 데이터를 바탕으로 임베딩 벡터를 만드는거야? 

아니다. 

Neural Collaborative Filtering (NCF) 모델은 기본적으로 사용자와 아이템의 상호작용 데이터(예: 사용자-아이템 평점, 클릭 기록 등)를 사용하여 임베딩 벡터를 학습합니다. 

### 실전 예제: 

PyTorch를 사용하여 MovieLens 데이터셋을 다루기 위한 커스텀 데이터셋 클래스를 정의합니다

이 클래스는 사용자가 정의한 데이터를 모델에 공급하기 쉽게 합니다. 

```python
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, mean_squared_error

# MovieLensDataset 클래스는 PyTorch의 Dataset 클래스를 상속받습니다.
class MovieLensDataset(torch.utils.data.Dataset):
    # 클래스의 초기화 메서드입니다. ratings라는 입력 데이터를 받습니다.
    def __init__(self, ratings):
        # ratings 데이터를 복사하고 numpy 배열로 변환합니다. 이렇게 함으로써 원본 데이터를 변경하지 않고 사용할 수 있습니다.
        data = ratings.copy().to_numpy()
        
        # 데이터의 첫 두 열(사용자와 아이템 ID)을 추출하여 정수형(int32)으로 변환한 후, ID를 0부터 시작하도록 1을 뺍니다.
        self.items = data[:, :2].astype(np.int32) - 1
        
        # 데이터의 세 번째 열(평점)을 추출하여 __preprocess_target 메서드를 통해 전처리한 후, 실수형(float32)으로 변환합니다.
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
        
        # 사용자와 아이템 ID의 최대값을 각각 찾고, 1을 더해 필드 차원을 설정합니다. 이는 사용자와 아이템의 수를 나타냅니다.
        self.field_dims = np.max(self.items, axis=0) + 1
        
        # 사용자 필드 인덱스를 numpy 배열로 설정합니다. 사용자 ID는 첫 번째 열에 위치하므로 0을 사용합니다.
        self.user_field_idx = np.array((0, ), dtype=np.int64)
        
        # 아이템 필드 인덱스를 numpy 배열로 설정합니다. 아이템 ID는 두 번째 열에 위치하므로 1을 사용합니다.
        self.item_field_idx = np.array((1,), dtype=np.int64)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target = target / 5.
        return target
```

이 코드는 PyTorch를 사용하여 다층 퍼셉트론(MLP)을 정의한 것입니다.

MLP는 신경망의 한 종류로, 입력층과 출력층 사이에 하나 이상의 은닉층이 있는 구조입니다. 각 은닉층은 선형 변환(Linear), 정규화(BatchNorm), 활성화 함수(ReLU), 드롭아웃(Dropout) 등을 포함합니다.

```python
# PyTorch의 Module 클래스를 상속받아 MLP 클래스를 정의합니다.
class MultiLayerPerceptron(torch.nn.Module):
    # 클래스의 초기화 메서드입니다. 여러 파라미터를 받습니다.
    # input_dim: 입력 차원
    # embed_dims: 각 은닉층의 차원 리스트
    # dropout: 드롭아웃 비율
    # output_layer: 출력 레이어 포함 여부
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        # 부모 클래스의 초기화 메서드를 호출합니다.
        super().__init__()
        
        # 레이어를 담을 리스트를 초기화합니다.
        layers = list()
        
        for embed_dim in embed_dims:
            # 선형 변환 레이어를 추가합니다. (입력 차원 -> 은닉층 차원)
            # 목적: 선형 변환 레이어(torch.nn.Linear)는 입력 데이터를 다음 층으로 전달하기 위한 변환을 수행합니다. 각 레이어는 입력 데이터의 특정 패턴을 학습하고 이를 변환하여 다음 레이어로 전달합니다.
            # 다중 레이어: 여러 개의 선형 변환 레이어를 사용함으로써 모델의 표현력을 높일 수 있습니다. 이는 더 복잡한 패턴을 학습하고, 데이터의 다양한 특성을 효과적으로 포착할 수 있게 합니다.
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            
            # 배치 정규화 레이어를 추가합니다. (은닉층 차원)
            # 목적: 배치 정규화(torch.nn.BatchNorm1d)는 각 미니 배치에 대해 입력을 정규화하여 학습을 안정화하고 가속화합니다.
            # 역할: 배치 정규화는 미니 배치 단위로 입력 데이터의 평균과 분산을 정규화합니다. 이를 통해 학습 중의 내부 공변량 이동(Internal Covariate Shift)을 줄여줍니다.
            # 학습 안정화: 배치 정규화를 사용하면 학습 속도를 높이고, 더 높은 학습률을 사용할 수 있습니다.
            # 과적합 방지: 정규화 과정에서 약간의 노이즈가 추가되기 때문에, 이는 과적합을 방지하는 데 도움을 줄 수 있습니다.
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            
            # ReLU 활성화 함수를 추가합니다.
            layers.append(torch.nn.ReLU())
            
            # 드롭아웃 레이어를 추가합니다. (드롭아웃 비율)
            # 목적: 드롭아웃(torch.nn.Dropout)은 학습 과정에서 무작위로 뉴런을 비활성화하여 모델이 과적합되는 것을 방지합니다.
            # 역할: 드롭아웃 레이어는 각 학습 단계에서 주어진 확률(p=dropout)로 뉴런을 비활성화합니다. 이는 뉴런들이 상호 의존하지 않고 독립적으로 학습되도록 만듭니다.
            # 과적합 방지: 드롭아웃은 학습 중에 네트워크가 특정 뉴런이나 경로에 과도하게 의존하지 않도록 합니다. 이는 모델이 더 일반화된 패턴을 학습하게 도와줍니다.
            layers.append(torch.nn.Dropout(p=dropout))
            
            # 다음 레이어의 입력 차원을 현재 레이어의 출력 차원으로 업데이트합니다.
            input_dim = embed_dim
        
        if output_layer:
            # 출력 레이어를 추가합니다. (마지막 은닉층 차원 -> 1)
            layers.append(torch.nn.Linear(input_dim, 1))
        
        # nn.Sequential을 사용하여 레이어를 순차적으로 묶어줍니다.
        self.mlp = torch.nn.Sequential(*layers)

    # 입력 x를 순차적으로 정의된 레이어들을 통과시켜 최종 출력을 반환합니다.
    def forward(self, x):
        return self.mlp(x)
```

FeaturesEmbedding 클래스를 정의한 것으로, 이 클래스는 Neural Collaborative Filtering(NCF)에서 사용자와 아이템의 임베딩을 계산하는 데 사용됩니다

```python
# torch.nn.Module을 상속받아 FeaturesEmbedding 클래스를 정의합니다. 이는 PyTorch의 모든 신경망 모듈의 기본 클래스입니다.
class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        # 클래스의 초기화 메서드입니다. field_dims와 embed_dim을 인자로 받습니다.
        # super().__init__()를 호출하여 부모 클래스(torch.nn.Module)의 초기화 메서드를 호출합니다.
        super().__init__()
        
        # torch.nn.Embedding을 사용하여 임베딩 레이어를 정의합니다. sum(field_dims)는 모든 필드의 차원을 합친 값으로, 이는 임베딩 레이어의 입력 차원이 됩니다. embed_dim은 임베딩 벡터의 차원입니다.
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        
        # self.offsets는 각 필드의 시작 인덱스를 계산하여 저장하는 배열입니다.
        # np.cumsum(field_dims)[:-1]는 각 필드의 누적 합을 계산한 후 마지막 요소를 제외한 배열입니다.
        # *는 배열을 언패킹하여 (0, ...) 형태의 튜플로 만듭니다.
        # 결과적으로 self.offsets는 각 필드가 시작되는 위치를 나타내는 배열이 됩니다.
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        
        # torch.nn.init.xavier_uniform_을 사용하여 임베딩 레이어의 가중치를 Xavier 초기화 방식으로 초기화합니다. 이는 가중치가 적절하게 초기화되어 학습이 잘 이루어지도록 합니다.
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)
```

Neural Collaborative Filtering 클래스: 
- NCF 클래스는 추천 시스템에 사용되는 모델로서 사용자가 아이템에 매긴 평점을 학습해서 사용자가 얼마의 평점을 매길지 알 수 있도록 한다. 이 과정에서 사용자 벡터와 아이템 벡터를 계산하는 방법도 학습함. 

```python
# torch.nn.Module을 상속받아 NeuralCollaborativeFiltering 클래스를 정의합니다.
class NeuralCollaborativeFiltering(torch.nn.Module):
  
    # 초기화 메서드입니다. 필요한 매개변수들을 인자로 받습니다.
    # field_dims: 각 필드의 차원 수
    # user_field_idx: 사용자 필드의 인덱스
    # item_field_idx: 아이템 필드의 인덱스
    # embed_dim: 임베딩 벡터의 차원
    # mlp_dims: 다층 퍼셉트론(MLP)의 각 레이어 차원
    # dropout: 드롭아웃 비율
    def __init__(self, field_dims, user_field_idx, item_field_idx, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.user_field_idx = user_field_idx
        self.item_field_idx = item_field_idx
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        
        # 임베딩 레이어의 출력 차원을 계산합니다. 이는 각 필드의 임베딩 벡터를 모두 합친 차원입니다.
        self.embed_output_dim = len(field_dims) * embed_dim
        
        # MultiLayerPerceptron 클래스를 사용하여 MLP를 초기화합니다.
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        
        # MLP의 최종 출력과 GMF의 출력을 결합하여 하나의 스칼라 값을 출력하는 선형 레이어를 정의합니다.
        self.fc = torch.nn.Linear(mlp_dims[-1] + embed_dim, 1)

    # 순전파(Forward) 메서드를 정의합니다. 입력 데이터 x를 받아 출력값을 반환합니다.
    def forward(self, x):
        # 입력 데이터를 임베딩 벡터로 변환합니다.
        x = self.embedding(x)
        
        # 사용자와 아이템 임베딩 벡터를 추출합니다.
        # squeeze(1)을 사용하여 불필요한 차원을 제거합니다.
        user_x = x[:, self.user_field_idx].squeeze(1)
        item_x = x[:, self.item_field_idx].squeeze(1)
        
        # 임베딩 벡터를 MLP에 입력하기 위해 적절한 차원으로 변환합니다.
        x = self.mlp(x.view(-1, self.embed_output_dim))
        
        # Generalized Matrix Factorization (GMF)를 계산합니다. 이는 사용자와 아이템 임베딩 벡터의 요소별 곱입니다.
        # GMF와 MLP 출력을 결합합니다.
        gmf = user_x * item_x
        x = torch.cat([gmf, x], dim=1)
        
        # 결합된 벡터를 최종 선형 레이어에 통과시켜 하나의 스칼라 값을 출력합니다.
        # squeeze(1)을 사용하여 불필요한 차원을 제거합니다.
        x = self.fc(x).squeeze(1)
        return torch.sigmoid(x)

```


### Q) Torch 의 Dataset:

Dataset 은 DataLoader 와 같이 써서, 모델에 필요한 데이터를 공급하기 위해서 쓰는거임

데이터셋 클래스를 통해 데이터 로딩을 구조화하고, 모델 학습에 필요한 데이터에 쉽게 접근할 수 있게 합니다. 이 클래스를 상속받아 커스텀 데이터셋을 정의할 수 있으며, 이는 특히 복잡한 데이터 전처리 과정을 처리하거나 다양한 형태의 데이터를 사용하는 경우 유용합니다.

PyTorch의 Dataset 클래스 설명
- torch.utils.data.Dataset 클래스는 다음과 같은 세 가지 주요 메서드를 가지고 있습니다:
- `__len__(self):`
  - 데이터셋의 크기를 반환합니다. 즉, 데이터셋에 몇 개의 데이터 포인트가 있는지를 나타냅니다.
  - 예를 들어, 데이터셋에 100개의 샘플이 있다면, __len__ 메서드는 100을 반환합니다.

- `__getitem__(self, index):`
  - 주어진 인덱스에 해당하는 데이터를 반환합니다.
  - 이 메서드는 데이터셋의 특정 위치에 있는 데이터를 로드하고 반환합니다. 인덱싱을 통해 데이터를 쉽게 접근할 수 있게 합니다.
  - 예를 들어, __getitem__(self, 0)는 데이터셋의 첫 번째 데이터를 반환합니다.

- `초기화 메서드 (__init__(self, ...)):`
  - 데이터셋 객체를 초기화할 때 사용됩니다.
  - 데이터셋의 경로, 데이터 파일, 기타 필요한 초기 설정 등을 정의할 수 있습니다.


사용 예: 
- 이 커스텀 데이터셋을 사용하여 DataLoader와 함께 사용할 수 있습니다. DataLoader는 배치 처리, 데이터 섞기(shuffling), 병렬 데이터 로딩 등의 기능을 제공합니다.

```python
from torch.utils.data import DataLoader

# 예시 데이터셋
data = [i for i in range(100)]

# 커스텀 데이터셋 객체 생성
dataset = CustomDataset(data)

# DataLoader 객체 생성
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# DataLoader를 사용하여 데이터 배치 반복
for batch in dataloader:
    print(batch)
```

### Torch 의 DataLoader 

Dataset 객체에서 데이터를 배치 단위로 로드하고, 데이터 섞기(shuffling), 병렬 처리 등의 기능을 제공합니다.


### Torch 의 다층 퍼셉트론(MLP) 이란? 

MLP 는 딥러닝 모델을 설계할 때 쓰는 하나의 신경망 중 하나로 가장 기본적인 신경망이다. 

이것 외에도 CNN, RNN, Transformer, LSTM 등이 있음. 

인공 신경망(Artificial Neural Network, ANN)의 한 종류로, 입력층과 출력층 사이에 하나 이상의 은닉층(Hidden Layer)을 포함하는 신경망 구조입니다. 

MLP는 비선형 데이터의 분류 및 회귀 문제를 해결하는 데 유용합니다. MLP는 퍼셉트론(Perceptron)이라고 불리는 기본 단위로 구성되며, 각 퍼셉트론은 입력 값을 받아 가중합을 계산하고 활성화 함수를 통해 출력 값을 생성합니다.

MLP의 구성 요소
- 입력층 (Input Layer):
  - 외부 데이터가 네트워크로 들어오는 층입니다.
  - 각 노드는 하나의 입력 피처(feature)를 나타냅니다.

- 은닉층 (Hidden Layers):
  - 입력층과 출력층 사이에 위치하며, 하나 이상의 층이 있을 수 있습니다.
  - 각 노드는 이전 층의 출력 값을 받아 가중합을 계산한 후 활성화 함수를 적용합니다.
  - 은닉층의 노드 수와 층 수는 신경망의 복잡도와 성능에 영향을 미칩니다.

- 출력층 (Output Layer):
  - 네트워크의 최종 출력이 생성되는 층입니다.
  - 회귀 문제에서는 일반적으로 하나의 노드를 가지며, 분류 문제에서는 클래스 수에 따라 여러 노드를 가질 수 있습니다.

- 가중치 (Weights)와 편향 (Biases):
  - 각 연결에는 가중치가 할당되며, 각 노드에는 편향 값이 추가됩니다.
  - 학습 과정에서 가중치와 편향이 조정되어 모델이 최적화됩니다.

- 활성화 함수 (Activation Functions):
  - 각 노드의 출력 값을 비선형 변환하여 다음 층으로 전달합니다.
  - 일반적인 활성화 함수로는 ReLU(Rectified Linear Unit), Sigmoid, Tanh 등이 있습니다.

MLP의 장점과 한계
- 장점:
  - 설계와 구현이 비교적 간단합니다.
  - 다양한 회귀 및 분류 문제에 적용 가능합니다.
  - 비선형성을 학습할 수 있는 능력이 있습니다.

- 한계:
  - 공간적(local) 구조를 갖는 데이터(예: 이미지)에는 비효율적입니다.
  - 시퀀스 데이터(예: 시간 순서가 있는 데이터) 처리에 한계가 있습니다.
  - 깊은 네트워크(많은 층)에서의 학습이 어려울 수 있습니다(즉, 깊이가 깊어질수록 기울기 소실 문제 발생 가능).

### Q) Torch 의 nn.Sequential 이란? 그리고 레이어를 순차적으로 묶는다는 의미는 

nn.Sequential은 PyTorch에서 제공하는 컨테이너 모듈로, 여러 개의 신경망 레이어를 순차적으로 묶어 하나의 모듈로 만드는 데 사용됩니다. 

이를 통해 모델의 각 레이어를 차례대로 정의하고 실행할 수 있습니다. nn.Sequential을 사용하면 코드가 더 깔끔하고 간결해지며, 레이어를 순서대로 연결하는 작업이 쉬워집니다

레이어를 순차적으로 묶는다는 것은, 입력 데이터가 첫 번째 레이어를 통과한 후 그 출력이 다음 레이어로 전달되고, 다시 그 출력이 다음 레이어로 전달되는 식으로 각 레이어를 차례로 통과하는 것을 의미합니다. 이러한 방식으로 여러 레이어를 하나의 큰 네트워크로 결합할 수 있습니다.

리스트로 입력한 여러개의 레이어를 하나로 결합한다는게 레이어를 순차적으로 묶는다는거임.  


### Q) 레이어에서 선형 변환을 넣는게 일반적이야? 선형 변환은 그리고 데이터를 추상화해서 표현한다고 알면 되는거지?

레이어에서 선형 변환을 넣는 것은 딥러닝 모델에서 매우 일반적입니다. 

선형 변환 레이어(torch.nn.Linear 등)는 신경망의 기본 구성 요소 중 하나로, 데이터를 추상화하여 표현하는 데 중요한 역할을 합니다.


### Q) torch.nn.Module 을 신경망 모듈이라고 했잖아. 이걸 가지고 기본적으로 딥러닝 신경망을 구축하는게 일반적이야? 

torch.nn.Module은 PyTorch에서 신경망을 구축하는 기본 단위입니다. 딥러닝 신경망을 구축할 때 매우 일반적으로 사용됩니다.

torch.nn.Module을 상속받아 자신만의 신경망 구조를 정의하고, 이를 통해 다양한 딥러닝 모델을 구축할 수 있습니다

torch.nn.Module의 중요성: 
- 모듈화:
  - torch.nn.Module은 딥러닝 모델의 구성 요소(모듈)를 정의하는 데 사용됩니다. 이를 통해 신경망을 모듈화하고 재사용할 수 있습니다.
  - 각 모듈은 하나의 레이어 또는 레이어의 집합을 나타낼 수 있으며, 이를 조합하여 복잡한 신경망을 만들 수 있습니다.

- 계층 구조:
  - torch.nn.Module을 상속받아 정의된 클래스는 다른 모듈을 포함할 수 있습니다. 이를 통해 계층적인 신경망 구조를 쉽게 구축할 수 있습니다.
  - 예를 들어, 전체 모델은 여러 개의 서브 모듈을 포함하고, 각 서브 모듈은 다시 여러 레이어를 포함할 수 있습니다.

### Q) 신경망에서는 임베딩 레이어라는게 있어? 이게 뭐야? 그리고 이건 입력 레이어 은닉층 레이어와 같은 하나의 레이어인거야?

신경망에서는 임베딩 레이어(Embedding Layer)라는 것이 있으며, 이는 특히 자연어 처리(NLP)와 추천 시스템에서 자주 사용되는 레이어입니다

임베딩 레이어는 입력 데이터를 고차원 공간에서 저차원 공간으로 매핑하는 역할을 합니다. 이는 이산적이고 고차원의 입력 데이터를 연속적이고 저차원인 밀집 벡터로 변환하여, 신경망이 더 효율적으로 학습할 수 있도록 돕습니다.

임베딩 레이어는 이산적인 카테고리형 데이터를 밀집 벡터(dense vector)로 변환하는 레이어입니다. 예를 들어, 단어, 사용자 ID, 아이템 ID 등의 범주형 데이터를 고차원 공간에서 저차원 벡터 공간으로 매핑합니다.


### Q) 왜 임베딩 레이어는 범주형 데이터를 처리하기 위한 특수한 레이어야? 나는 자연어 같은 고차원 데이터를 저차원으로 분해해주는 레이어라고 생각했는데

범주형 데이터 자체가 고차원이라고 생각해도 되는거임. 
- 범주형 데이터는 고유한 값들로 이루어진 데이터입니다. 예를 들어, 단어, 사용자 ID, 제품 ID 등이 있습니다.

범주형 데이터를 고차원 벡터로 변환하는 가장 기본적인 방법은 원-핫 인코딩임. 근데 이건 메모리 효율적이지 않고, 빠르지도 않다. 그래서 임베딩을 이용하는거임.

원-핫 인코딩 데이터 표현: 
```text
// 단어 사이즈 5개 기준
One-hot encoding of 'apple': [1. 0. 0. 0. 0.]
One-hot encoding of 'banana': [0. 1. 0. 0. 0.]
```

임베딩 표현: 
```text
Embedding of 'apple': [ 0.43612888 -0.48714092  0.12159146]
Embedding of 'banana': [-0.21120584  0.49931687  0.03114444]
```

### Q) FeaturesEmbedding 에서 self.embeddings 에 대해 정리 

self.embedding 은 torch.nn.Embeddings 로 생성됨. torch.nn.Embeddings 는 임베딩 계산을 해줘서 벡터로 만들어주는 레이어 역할을 한다. 

즉 고차원 데이터를 저차원인 임베딩 벡터로 변환시켜주는 역할을 하는거지. 

`self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)` 이렇게 만들 때 `sum(filed_dims)` 로 필드 차원의 합을 넣는다. 
- 여기서 필드는 범주형 데이터들을 말함. 사용자 Id, 아이템 Id, 카테고리 Id 등등 
- field_dims 는 범주형 데이터의 차원을 말한다. 그러니까 사용자가 1000명, 아이템이 500개, 카테고리가 20개가 있다면 범주형 데이터 차원은 이렇게 표현됨 \[1000, 500, 20]
- sum(field_dims) 는 필드의 차원을 합친 걸 말한다. 기존에 필드의 차원이 \[1000, 500, 20] 이 었는데 이 차원을 다 합치면 1520개의 범주가 있는거임. 1520 개의 범주를 처리하는게 필요한거지. 그리고 이렇게 만드는 걸 임베딩 테이블에서 모든 필드의 범주 데이터를 처리하게 만드는 거라고 함. 
- 1520개의 차원을 처리하기 위해서 임베딩 벡터의 크기를 1520으로 맞출 필요는 없다. embed_dim 으로 정의해서 임베딩 벡터의 크기를 정의할 수 있음. 여기서는 50으로 정의한다. 50개의 벡터로 1520개의 차원을 표현할 수 있는거. 

### Q) FeaturesEmbedding 에서 self.offsets 에 대한 정리

코드 전체: 
```python
class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)
```

`self.offsets:` 
- 배열을 생성하여 각 필드가 시작되는 위치를 나타낸다. 
- 그러니까 field_dims가 \[1000, 500, 20] 라고 가정해보자.
- 이 경우에 np.cumsum(field_dims) 으로 누적 합을 계산하면 \[1000, 1500, 1520] 을 반환하게 되고, 여기서 `[:-1]` 로 인해 마지막 요소를 제외하면 \[1000, 1500] 가 된다. 그리고 이를 배열로 만들면 최종적으로 \[0, 1000, 1500] 이 된다. 
- 이 \[0, 1000, 1500] 는 각 필드가 시작되는 인덱스를 말하게 됨. 첫 번째 범주는 0번째 인덱스에서 시작하고, 두 번째 범주는 1000번째 인덱스에서 시작하고, 세 번째 범주는 1500번째 인덱스에서 시작함을 말한다. 
- 즉 모델이 각 입력 필드의 시작 위치를 알게 되면서 필드를 임베딩으로 변환하는데 사용된다. 


self.offsets 는 어떻게 활용되는가? 
- 예시로 field_dims = \[1000, 500, 20] 라고 가정해보자. 
- 입력 데이터로는 아래와 같이 각 범주에 해당하는 숫자로 들어온다. 
- 그러니까 \[999, 499,  19] 라고 데이터가 들어오면 (사용자 Id: 999, 아이템 id: 499, 카테고리 아이디: 19) offset 을 통해서 다음으로 \[999, 1499, 1519] 변환될거임. 그리고 \[999, 1499, 1519] 가 임베딩으로 변환된다.
- 이렇게 offset 을 더해서 정확한 위치를 찾는 이유로는 사용자, 아이템, 카테고리 세 가지 범주형 데이터가 합쳐져서 하나의 큰 임베딩 테이블 (1520개) 로 관리되고 있어서임. 이를 통해 각 범주형 데이터들은 임베딩 테이블에서 표현될 수 있는거다. 

```python
# 예시 입력 데이터 (사용자 ID, 아이템 ID, 카테고리 ID)
input_data = torch.LongTensor([[0, 0, 0], [999, 499, 19]])  # 사용자 ID 0, 999; 아이템 ID 0, 499; 카테고리 ID 0, 19
```


### FeatureEmbedding 의 해당 코드에 대해 좀 더 자세하게 `torch.nn.init.xavier_uniform_(self.embedding.weight.data)` 

먼저 임베딩 레이어에 대해 알아야한다: 
- 임베딩 레이어는 torch.nn.Embedding 에 의해 구현되며 FeatureEmbedding 에서는 self.embedding 에 저장되는 값이다. 
- self.embedding 여기에는 각 범주형 데이터의 크기들이 하나로 합쳐진 임베딩 테이블이 있고, 임베딩 테이블은 하나의 배열과 같이 이뤄져서 범주형 데이터를 표현할 수 있는데, 이 범주형 데이터들은 임베딩 레이어에 의해서 임베딩 벡터로 변환된다.  
- 임베딩 벡터는 학습 가능한 파라미터임. 

그다음 임베딩 레이어 가중치에 대해 알아야한다:
- 임베딩 레이어 가중치는 임베딩 테이블에 있는 행을 임베딩 벡터를 만들 때 사용되는 가중치임. 

이제 Xavier 초기화 방식에 대해 알아보자: 
- 이건 임베딩 레이어의 가중치를 초기화 하는 방법임. 목적은 학습을 더 잘 하도록 학습의 시작점을 결정하기 위해서임.

Xavier 초기화가 왜 더 학습을 잘하게 만드는건가? 
- 가중치를 적절하게 초기화해서 그라디언트 소실 문제와 그라디언트 폭발이 잘 안일어나게 만드는 방법이라서 그럼. 
- Xavier 초기화는 입력과 출력의 노드 수를 고려하여 가중치를 초기화함. 입출력 노드 수가 많을수록 연쇄 룰에 의해서 그라디언트가 커지거나 작아질 수 있음. 


### NCF 클래스의 해당 코드에 대해 좀 더 자세하게 `self.embed_output_dim = len(field_dims) * embed_dim`: 

먼저 임베딩 레이어 출력에 대해 알아야한다: 
- 임베딩 레이어의 출력은 범주형 데이터들 (e.g 사용자, 아이템, 카테고리) 등이 임베딩 벡터로 변환되서 출력된다. 
- 이때 벡터의 차원 개수는 embed_dim 의 값으로 정의된다. 

임베딩 레이어의 출력 차원은 각 범주형 데이터의 임베딩 벡터에다가 범주형 데이터의 수 (= 즉 필드 차원의 수) 로 정해진다. 
- 그러니까 범주형 데이터가 3개라면 (사용자, 아이템, 카테고리) 3개의 임베딩 벡터가 출력되야하는거임.
- 그래서 `self.embed_output_dim = len(field_dims) * embed_dim` 로 계산되는거
- 이 `self.embed_output_dim = len(field_dims) * embed_dim` 값은 임베딩 벡터가 필드 차원의 수만큼 출력되는 걸 말하는거임.


### NCF 클래스의 해당 코드에 대해 좀 더 자세하게 `self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)`: 

self.mlp 에다가 레이어 구조를 넣어주는거임. 아래의 `MultiLayerPerceptron(torch.nn.Module)` 와 `NeuralCollaborativeFiltering(torch.nn.Module)` 를 참고하면 된다.

이렇게 레이어 구조를 넣어주면서 모듈을 조립해서 사용할 수 있는거임. 

`MultiLayerPerceptron(torch.nn.Module):`

```python
class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
```

`NeuralCollaborativeFiltering(torch.nn.Module)`

```python
class NeuralCollaborativeFiltering(torch.nn.Module):
    def __init__(self, field_dims, user_field_idx, item_field_idx, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.user_field_idx = user_field_idx
        self.item_field_idx = item_field_idx
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.fc = torch.nn.Linear(mlp_dims[-1] + embed_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        user_x = x[:, self.user_field_idx].squeeze(1)
        item_x = x[:, self.item_field_idx].squeeze(1)
        x = self.mlp(x.view(-1, self.embed_output_dim))
        gmf = user_x * item_x
        x = torch.cat([gmf, x], dim=1)
        x = self.fc(x).squeeze(1)
        return torch.sigmoid(x)
```

### NeuralCollaborativeFiltering 클래스의 forward 메소드에 대해서 좀 더 자세하게 

순전파 과정에 대해 먼저 복기해보자: 
- 입력 데이터가 임베딩 레이어를 통과한 이후 예측 평점 값을 생성하는 단계다. 
- 그리고 이 과정에서 MLP 와 GMF 를 계산한 후 최종 출력을 생성한다. 

```python
def forward(self, x):
        x = self.embedding(x)
        user_x = x[:, self.user_field_idx].squeeze(1)
        item_x = x[:, self.item_field_idx].squeeze(1)
        x = self.mlp(x.view(-1, self.embed_output_dim))
        gmf = user_x * item_x
        x = torch.cat([gmf, x], dim=1)
        x = self.fc(x).squeeze(1)
        return torch.sigmoid(x)
```
