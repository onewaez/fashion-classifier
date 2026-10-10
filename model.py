import numpy as np

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, current_data):
        for layer in self.layers:
            current_data = layer.forward(current_data)
        return current_data

    def backward(self, current_gradient, learning_rate):
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient, learning_rate)

    def save_weights(self, filename):
        weights_dict = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                weights_dict[f'layer_{i}_weights'] = layer.weights
                weights_dict[f'layer_{i}_biases'] = layer.biases
        np.savez(filename, **weights_dict)

    def load_weights(self, filename):
        weights_dict = np.load(filename)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                layer.weights = weights_dict[f'layer_{i}_weights']
                layer.biases = weights_dict[f'layer_{i}_biases']