# Maximum Likelihood Estimation 


Maximum Likelihood Estimation (MLE) is a method used to estimate the parameters of a statistical model.

Step 1: Understand the Goal
- The goal of MLE is to find the parameter values that make the observed data most likely under the assumed statistical model.


Step 2: Define the Model
- Assume you have a statistical model with a probability distribution. For example, suppose you believe your data comes from a normal (Gaussian) distribution with mean 𝜇 μ and standard deviation σ.

![](./images/MLE%20Step%203.png)
![](./images/MLE%20Step%204.png)
![](./images/MLE%20Step%205.png)
![](./images/MLE%20Step%206.png)


#### Q) What is the parameters of a statistical model

The parameters of a statistical model are the quantities that define the behavior and characteristics of the model. These parameters are typically estimated from data and are used to make predictions, understand relationships, and infer properties about the population from which the data is drawn.

![](./images/statistical%20model%20parameter.png)


#### Q) What is the likelihood function?

The likelihood function is a fundamental concept in statistical inference, particularly in the method of maximum likelihood estimation (MLE). It measures how likely it is to observe the given sample data under different parameter values of a statistical model. Here is a detailed explanation of the likelihood function:

Purpose and Use: 
- Model Comparison: The likelihood function can also be used to compare different statistical models. Models that provide a higher likelihood for the observed data are generally preferred.


#### Q) 딥러닝에서 Maximum likelihood estimation 에 대해 알아야하는 이유는?

손실 홤수의 최적화 과정과 MLE 는 밀접한 관련이 있음. 

손실 함수를 최적화 하는 과정이 가지는 의미는 예측을 잘하도록 하는 의미임. MLE 를 통해 올바른 확률 분포를 찾는 과정은 데이터가 이 확률 분포에서 나왔을 확률이 높음을 나타내고, 이걸 통해 분류 작업을 잘 할 수 있다는 의미가 되기도 한다. 

