import numpy as np

# Dataset
x = np.array([1, 4, 7, 9, 12, 25, 26, 30])
y = np.array([2, 6, 10, 13, 23, 35, 40, 90])

# Initial values
b0 = 0
b1 = 0
alpha = 0.0001  # Reduced learning rate
iterations = 50000  # Reduced number of iterations

# Gradient Descent Loop
for i in range(iterations):
    y_pred = b0 + b1 * x
    error = y - y_pred
    
    b0 += alpha / len(x) * np.sum(error)
    b1 += alpha / len(x) * np.sum(error * x)

print(f"Error = {np.sum(error**2)}")
print(f"Model: y = {b0} + {b1} * x")

# Prediction
x_input = float(input("Enter the value of x you want to predict: "))
y_output = b0 + b1 * x_input
print(f"Predicted value of y when x = {x_input} is {y_output}")


# Question no 2
# from sklearn.linear_model import LinearRegression
# import numpy as np

# x = np.array([500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]).reshape(-1, 1)  # Reshape for sklearn
# y = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95])

# model = LinearRegression()

# model.fit(x, y)

# X = int(input("Enter the value of x you want to predict: "))
# y_pred = model.predict(np.array([[X]]))

# print(f"Predicted value of y when x = {X} is {y_pred}")


#question no 3
# from sklearn.linear_model import LogisticRegression
# import numpy as np

# x = np.array([1, 4, 7, 10, 14, 18, 23, 28, 32, 36]).reshape(-1, 1)
# y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0])

# model = LogisticRegression()
# model.fit(x, y)

# X = int(input("Enter the value of x you want to predict: "))
# y_pred = model.predict(np.array([[X]]))

# print(f"Predicted value of y when x = {X} is {y_pred[0]}")


# Question no 4
# import numpy as np

# class HebbianLearning:
#     def __init__(self, input_size, learning_rate=0.1):
#         self.weights = np.random.rand(input_size)
#         self.bias = np.random.rand()
#         self.learning_rate = learning_rate

#     def predict(self, x):
#         activation = np.dot(x, self.weights) + self.bias
#         return activation

#     def train(self, x, target):
#         prediction = self.predict(x)
#         for i in range(len(x)):
#             self.weights[i] += self.learning_rate * x[i] * target
#         self.bias += self.learning_rate * target

# if __name__ == "__main__":
#     input_size = 3
#     model = HebbianLearning(input_size)
#     training_data = np.array([[0.1, 0.2, 0.3],
#                                [0.2, 0.1, 0.4],
#                                [0.4, 0.5, 0.6]])
#     targets = np.array([1, 0, 1])
#     for epoch in range(5):
#         for x, target in zip(training_data, targets):
#             model.train(x, target)
#             print(f"Updated weights: {model.weights}, Updated bias: {model.bias}")

#     test_input = np.array([1, 1.5, 1.6])
#     prediction = model.predict(test_input)
#     print(f"Prediction for input {test_input}: {prediction}")

# Question no 5
# import numpy as np
# class Perceptron:
#     def __init__(self, input_size, learning_rate=0.1):
#         # Initialize weights and bias
#         self.weights = np.zeros(input_size)
#         self.bias = 0
#         self.learning_rate = learning_rate

#     def predict(self, x):
#         weighted_sum = np.dot(x, self.weights) + self.bias
#         return 1 if weighted_sum >= 0 else 0

#     def train(self, training_data, targets, epochs):
#         for epoch in range(epochs):
#             for x, target in zip(training_data, targets):
#                 prediction = self.predict(x)
#                 error = target - prediction
#                 self.weights += self.learning_rate * error * x
#                 self.bias += self.learning_rate * error

# if __name__ == "__main__":
#     training_data = np.array([[0, 0],
#                                [0, 1],
#                                [1, 0],
#                                [1, 1]])
#     targets = np.array([0, 0, 0, 1])

#     perceptron = Perceptron(input_size=2)

#     perceptron.train(training_data, targets, epochs=10)
#     print("Testing the Perceptron on AND gate:")
#     for x in training_data:
#         output = perceptron.predict(x)
#         print(f"Input: {x}, Predicted Output: {output}")


# Question no 6
# import numpy as np

# class NeuralNetwork:
#     def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
#         self.learning_rate = learning_rate
#         self.weights_input_hidden = np.random.rand(input_size, hidden_size)
#         self.weights_hidden_output = np.random.rand(hidden_size, output_size)
#         self.bias_hidden = np.random.rand(hidden_size)
#         self.bias_output = np.random.rand(output_size)

#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))

#     def sigmoid_derivative(self, x):
#         return x * (1 - x)

#     def forward(self, x):
#         self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
#         self.hidden_output = self.sigmoid(self.hidden_input)
#         self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
#         self.final_output = self.sigmoid(self.final_input)
#         return self.final_output

#     def backward(self, x, y, output):
#         output_error = y - output
#         output_delta = output_error * self.sigmoid_derivative(output)

#         hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
#         hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

#         self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * self.learning_rate
#         self.bias_output += np.sum(output_delta, axis=0) * self.learning_rate
#         self.weights_input_hidden += np.dot(x.T, hidden_delta) * self.learning_rate
#         self.bias_hidden += np.sum(hidden_delta, axis=0) * self.learning_rate

#     def train(self, x, y, epochs):
#         for epoch in range(epochs):
#             output = self.forward(x)
#             self.backward(x, y, output)
            
#             if epoch % 5000 == 0:
#                 print(f"Epoch {epoch}:")
#                 print(f"Weights (Input to Hidden): \n{self.weights_input_hidden}")
#                 print(f"Weights (Hidden to Output): \n{self.weights_hidden_output}")
#                 print(f"Biases (Hidden): \n{self.bias_hidden}")
#                 print(f"Biases (Output): \n{self.bias_output}\n")

# if __name__ == "__main__":
#     training_data = np.array([[0, 0],
#                                [0, 1],
#                                [1, 0],
#                                [1, 1]])

#     targets = np.array([[0],
#                         [1],
#                         [1],
#                         [0]])
#     input_size = 2
#     hidden_size = 2
#     output_size = 1
#     nn = NeuralNetwork(input_size, hidden_size, output_size)
    
#     nn.train(training_data, targets, epochs=10000)

#     print("Testing the neural network on XOR problem:")
#     for x in training_data:
#         output = nn.forward(x)
#         print(f"Input: {x}, Predicted Output: {output[0]:.4f}")