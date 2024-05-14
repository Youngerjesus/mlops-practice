# Backpropagtion 

## Bakpropagation 원리 정리 

![](./images/Backward%20Process%201.png)
![](./images/Backward%20Process%202.png)
![](./images/Backward%20Process%203.png)
![](./images/Backward%20Process%204.png)
![](./images/Backward%20Process%205.png)
![](./images/Backward%20Process%206.png)


#### Show me the Python code of the backpropagation process.

```python
import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input data (4 samples, 3 features each)
X = np.array([[0, 0, 1],
              [1, 1, 1],
              [1, 0, 1],
              [0, 1, 1]])

# Output data (4 samples, 1 output each)
y = np.array([[0],
              [1],
              [1],
              [0]])

# Seed the random number generator for reproducibility
np.random.seed(1)

# Initialize weights randomly with mean 0
input_layer_neurons = X.shape[1]  # number of features in input data
hidden_layer_neurons = 4  # number of hidden layer neurons
output_neurons = 1  # number of neurons in output layer

# Initialize weights and biases
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))

bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Training parameters
learning_rate = 0.1
epochs = 10000

# Training the neural network
for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_activation = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_activation, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Calculate the error
    error = y - predicted_output
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1}, Error: {np.mean(np.abs(error))}")

    # Backpropagation
    # Calculate the derivative of the error with respect to the output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    # Calculate the error for the hidden layer
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)

    # Update weights and biases
    weights_hidden_output += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Output after training
print("Output after training:")
print(predicted_output)

```

#### Q) Please explain the code of the backpropagation process by connecting it with a mathematical formula

#### Q) Dot 연산은 뭔데? 

