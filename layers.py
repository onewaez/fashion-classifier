import numpy as np

class Linear:
    def __init__(self, n_inputs, n_neurons):
        # He/Kaiming initialization
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros(n_neurons)
        self.saved_input = None

    def forward(self, input_data):
        self.saved_input = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, gradient_from_next_layer, learning_rate):
        gradient_for_previous_layer = np.dot(gradient_from_next_layer, self.weights.T)
        gradients_for_weights = np.dot(self.saved_input.T, gradient_from_next_layer)
        gradients_for_biases = np.sum(gradient_from_next_layer, axis=0)
        
        self.weights -= learning_rate * gradients_for_weights
        self.biases -= learning_rate * gradients_for_biases
        
        return gradient_for_previous_layer

class ReLU:
    def __init__(self):
        self.saved_input = None

    def forward(self, input_data):
        self.saved_input = input_data
        return np.maximum(0, input_data)

    def backward(self, gradient_from_next_layer, learning_rate):
        gradient_for_previous_layer = np.array(gradient_from_next_layer, copy=True)
        gradient_for_previous_layer[self.saved_input <= 0] = 0
        return gradient_for_previous_layer