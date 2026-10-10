import numpy as np
from PIL import Image
from layers import Linear, ReLU
from model import Sequential

FASHION_CLASSES = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
])

model.load_weights("fashion_mnist_weights.npz")
print("Weights loaded successfully!")

def process_image(image_path):
    img = Image.open(image_path).convert('L')
    
    # Center crop to square aspect ratio
    width, height = img.size
    new_dim = min(width, height)
    left = (width - new_dim) / 2
    top = (height - new_dim) / 2
    right = (width + new_dim) / 2
    bottom = (height + new_dim) / 2
    img = img.crop((left, top, right, bottom))
    
    img = img.resize((28, 28))
    img_array = np.array(img)
    
    # Invert colors if background is light
    if img_array[0, 0] > 128:
        img_array = 255 - img_array
        
    # Threshold background noise
    img_array[img_array < 70] = 0
    
    return img_array.reshape(1, 784) / 255.0

my_image = process_image("my_photo.jpg")
logits = model.forward(my_image)

exp_logits = np.exp(logits - np.max(logits))
probabilities = exp_logits / np.sum(exp_logits)

predicted_class_index = np.argmax(probabilities)
confidence = probabilities[0][predicted_class_index] * 100

print("\nModel Prediction:")
print(f"Class: {FASHION_CLASSES[predicted_class_index]}")
print(f"Confidence: {confidence:.2f}%")