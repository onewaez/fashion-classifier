import numpy as np
import idx2numpy
from layers import Linear, ReLU
from model import Sequential
from losses import CrossEntropyLoss

X_train_raw = idx2numpy.convert_from_file('dataset/train-images-idx3-ubyte')
y_train = idx2numpy.convert_from_file('dataset/train-labels-idx1-ubyte')

X_train = X_train_raw.reshape(X_train_raw.shape[0], 784) / 255.0

model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
])
criterion = CrossEntropyLoss()

epochs = 25
batch_size = 32
learning_rate = 0.01
n_samples = X_train.shape[0]

print("Starting training...")
for epoch in range(epochs):
    indices = np.random.permutation(n_samples)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    epoch_loss = 0
    
    for i in range(0, n_samples, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]
        
        logits = model.forward(X_batch)
        loss = criterion.forward(logits, y_batch)
        epoch_loss += loss
        
        start_gradient = criterion.backward(logits, y_batch)
        model.backward(start_gradient, learning_rate)
        
    avg_loss = epoch_loss / (n_samples / batch_size)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

model.save_weights("fashion_mnist_weights.npz")
print("Training completed. Weights saved to fashion_mnist_weights.npz")