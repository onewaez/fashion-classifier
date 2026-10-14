# Fashion Classifier: Building a Neural Network from Scratch

## Project Overview

This repository presents a foundational implementation of a Multi-Layer Perceptron trained on the Fashion-MNIST dataset. The primary objective of this project is to demonstrate the internal mechanics of a artificial neural network by constructing every component manually from first principles. By refraining from high-level machine learning frameworks, this implementation exposes the underlying linear algebra, matrix calculus, and optimization techniques that drive modern deep learning architectures.

The technical documentation for this repository is structured into sequential Stages, guiding the reader through the complete life cycle of an artificial neural network. Stage 1 covers binary data parsing and tensor normalization. Subsequent stages detail parameter initialization, non-linear forward propagation, loss calculation, backpropagation, and real-world image inference.

## The Dataset

The Fashion-MNIST dataset was created by Zalando as a modern replacement for the classical MNIST handwritten digit dataset. It consists of 70,000 grayscale images with a resolution of 28x28 pixels divided into 10 distinct garment categories, such as T-shirts, trousers, coats, and sneakers. The dataset is partitioned into 60,000 training examples and 10,000 evaluation examples, serving as a standardized benchmark for computer vision classification models.

## Core Architectural Concept

The neural network operates as a sequential pipeline processing raw high-dimensional pixel inputs into class probabilities. High-dimensional image arrays are flattened into one-dimensional feature vectors and passed through fully connected transformation layers. Non-linear activation functions allow the network to construct complex boundaries between different garment classes. 

During the training phase, predicted outputs are evaluated against true target labels using a loss function. The error signal is then propagated backward through every layer using matrix derivatives and the chain rule of calculus. An optimization algorithm continuously adjusts internal weights and biases to minimize overall loss across training epochs.

## Expected Outputs

Upon completing the execution pipeline, the system yields two primary outputs:

First, the trained network parameters, including weight matrices and bias vectors for all linear layers, are serialized and exported to a compressed NumPy file (`fashion_mnist_weights.npz`). This persistent weight checkpoint allows the model to perform immediate evaluation without re-training.

Second, during inference, the pipeline accepts arbitrary user-provided photographs, normalizes their spatial and spectral dimensions, and outputs a normalized probability distribution across all 10 garment categories. The final output presents the top predicted class along with its corresponding confidence percentage.

## Core Learning Outcomes

By studying and executing this repository, the reader achieves a comprehensive understanding of core deep learning mechanics:

Understanding how high-dimensional image tensors are formatted, reshaped, and scaled for optimal numerical gradient flow.

Mastering the mathematical formulation of fully connected layers, including He/Kaiming variance scaling to prevent vanishing and exploding gradients.

Deriving analytical gradients for non-linear activations and cross-entropy loss functions using vectorized linear algebra.

Constructing a production-ready data preprocessing pipeline capable of adapting uncurated real-world photos to match standardized dataset characteristics.

## Required Environment and Dependencies

The project relies on a minimal set of foundational Python packages to keep external abstractions to a minimum:

Python 3 runtime environment serving as the base operational engine.

NumPy providing optimized C-compiled array structures and matrix operations required for manual tensor calculus.

Pillow (PIL) enabling file operations, spatial cropping, channel conversion, and noise filtering for custom input photographs.

idx2numpy facilitating direct byte parsing of the binary IDX file format native to the Fashion-MNIST dataset.

# Stage 1: Data Pipeline and Preprocessing

## Understanding the Dataset Architecture

Machine learning models require clean, predictable data representations to perform gradient-based optimization efficiently. In this project, we utilize the **Fashion-MNIST** dataset created by Zalando Research. It serves as a modern drop-in replacement for the classical MNIST handwritten digits dataset, presenting a more complex computer vision challenge while maintaining the exact same dimensional footprint.

The dataset contains 70,000 grayscale images categorized into 10 distinct garment classes, split into 60,000 training examples and 10,000 evaluation examples. On disk, the data is distributed in the binary **IDX** file format, compressed into `.ubyte` files. Specifically, `train-images-idx3-ubyte` stores the raw pixel matrices, while `train-labels-idx1-ubyte` contains the target class indices ranging from 0 to 9.

## Reading Binary Files with idx2numpy

The IDX file structure is a minimal binary header format designed to store multidimensional arrays efficiently. The first four bytes of an IDX file contain metadata, including magic numbers that define the data type and the number of dimensions, followed by integers that specify the size of each dimension. The remainder of the file contains a contiguous stream of raw unformatted bytes representing pixel values.

To load these binary payloads directly into memory without writing custom C-extensions or relying on slow Python file-reading loops, we utilize the `idx2numpy` library.

```python
import idx2numpy

X_train_raw = idx2numpy.convert_from_file('dataset/train-images-idx3-ubyte')
y_train = idx2numpy.convert_from_file('dataset/train-labels-idx1-ubyte')
```

Under the hood, `idx2numpy` parses the binary header to extract tensor dimensions and memory strides. It then maps the raw byte buffer directly into a contiguous C-ordered `numpy.ndarray`. This provides instant memory access and eliminates performance bottlenecks during initial data ingestion.

<img width="1920" height="1080" alt="Stage1PixelMatrix_ManimCE_v0 22 0" src="https://github.com/user-attachments/assets/6d95c0d0-1d7d-48a6-9ad4-af6a6c6b59c8" />


## Analyzing Initial Tensor Dimensions

Immediately after reading the files into memory, we must inspect the structure and data types of the loaded NumPy arrays to verify structural integrity.

```python
print("Raw images shape:", X_train_raw.shape)
print("Raw labels shape:", y_train.shape)
print("Pixel data type:", X_train_raw.dtype)
```

The image tensor `X_train_raw` yields a shape of `(60000, 28, 28)`. This represents a three-dimensional tensor containing 60,000 individual matrices, where each matrix consists of 28 rows and 28 columns of pixels. The element data type is `uint8`, which stands for an unsigned 8-bit integer. The target label array `y_train` yields a one-dimensional vector of shape `(60000,)`, where each element represents the ground-truth integer class index for the corresponding image.

## Vectorization and Spatial Flattening

Our custom neural network relies on fully connected dense layers (`Linear`). A dense layer calculates activations using standard matrix multiplication expressed as $Z = X \cdot W + b$. Because linear algebra operations in fully connected layers expect input features to be structured as a flat one-dimensional vector per sample, a two-dimensional `28 × 28` spatial grid cannot be processed directly without reshaping.

To feed two-dimensional pixel grids into a dense layer, we must flatten the spatial dimensions into a single continuous feature array containing 784 numbers (`28 × 28 = 784`).

```python
X_train = X_train_raw.reshape(X_train_raw.shape[0], 784)
```

The `reshape` method reorganizes the memory buffer using row-major ordering, also known as C-style layout. During this operation, the first row of 28 pixels occupies array indices 0 through 27, the second row occupies indices 28 through 55, and this sequence continues sequentially until all 784 elements are appended. The shape of `X_train` transitions from `(60000, 28, 28)` to a two-dimensional matrix of shape `(60000, 784)`. Each row in this matrix now represents an individual fashion item as a flat feature vector.

## Pixel Intensity Physics and Normalization

In the raw `uint8` format, each pixel value is encoded using 8 bits of memory, providing $2^8 = 256$ distinct brightness levels. A value of 0 corresponds to absolute darkness, which forms the solid black background in the Fashion-MNIST dataset. A value of 255 represents maximum brightness, forming the primary highlights of the garments, while values between 1 and 254 represent intermediate grayscale tones.

Passing raw unscaled values ranging from 0 to 255 directly into a neural network creates severe numerical instability. When an unscaled vector contains numbers as large as 255, the linear combination $Z = \sum (x_i \cdot w_i) + b$ produces extremely large activation values on the very first forward pass. These large values cause numerical overflow when evaluating the exponential functions within Softmax ($e^Z$). Furthermore, excessively large inputs generate massive gradient updates during backpropagation, causing optimization steps to overshoot and fail to converge.

To ensure stable optimization and uniform gradient steps, we normalize the raw pixel values linearly into a floating-point range between 0.0 and 1.0.

```python
X_train = X_train_raw.reshape(X_train_raw.shape[0], 784) / 255.0
```

Dividing the array by the floating-point literal `255.0` automatically casts the underlying NumPy array from 8-bit integers (`uint8`) to double-precision floating-point numbers (`float64`). Dark background pixels with a value of 0 map precisely to `0.0`, maximum white highlights map to `1.0`, and mid-tone gray pixels scale proportionally to values such as `0.5`. This keeps all input signals within a well-behaved range, preventing numerical overflow and stabilizing weight updates.

## Data Pipeline Transformation Summary

| Processing Step | Applied Code Operation | Input Shape | Output Shape | Data Type | Value Range |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. File Parsing** | `idx2numpy.convert_from_file()` | Binary File | `(60000, 28, 28)` | `uint8` | $[0, 255]$ |
| **2. Flattening** | `.reshape(60000, 784)` | `(60000, 28, 28)` | `(60000, 784)` | `uint8` | $[0, 255]$ |
| **3. Normalization** | `/ 255.0` | `(60000, 784)` | `(60000, 784)` | `float64` | $[0.0, 1.0]$ |

## Stage 2: Network Topology & He (Kaiming) Initialization

### Network Architecture and Layer Connections

Our neural network uses a feedforward architecture with two fully connected layers. The network processes flattened image vectors containing 784 pixel values and predicts one of 10 fashion categories.

The first transformation layer, `Linear(784, 128)`, takes the 784 input pixels and connects them to 128 hidden neurons. This hidden layer allows the network to combine individual pixel values into meaningful visual features, such as edges, outlines, and shapes. The second layer, `Linear(128, 10)`, takes those 128 hidden features and maps them to 10 output class scores, called logits. The class with the highest logit score represents the network final prediction.

Inside every dense layer, input features are multiplied by a weight matrix $W$ and added to a bias vector $b$. For a mini-batch of images $X$, the linear forward pass is computed as matrix multiplication:

$$Z = X \cdot W + b$$

In the first layer, input matrix $X$ has a shape of `(N, 784)`, weight matrix $W$ has a shape of `(784, 128)`, and bias vector $b$ has a shape of `(128,)`. Across both layers, the network contains 101,770 total trainable parameters.

<img width="1920" height="1080" alt="Stage2TopologyDiagram_ManimCE_v0 22 0" src="https://github.com/user-attachments/assets/75840ef9-4e7c-4c37-ab94-46aa6a7982e1" />


### Why Weight Initialization Matters

Before training begins, weight parameters must be initialized with starting values. Setting initial weights incorrectly prevents the neural network from learning effectively during training.

Initializing all weights to zero creates a problem called symmetry. When every weight starts at zero, every neuron in the hidden layer produces the exact same calculation and receives the exact same gradient update during backpropagation. Instead of 128 unique neurons learning different visual features, all neurons behave identically, reducing the layer to a single effective unit.

Randomly picking weight values without proper scaling also breaks model training. If initial weights are too large, multiplying 784 inputs by large numbers causes activation values to explode into massive numbers. This leads to unstable gradients during backpropagation. Conversely, if initial weights are too small, signals shrink toward zero as they pass through the layers, causing gradients to vanish and stopping learning entirely.

### How He (Kaiming) Initialization Works

To keep signals stable throughout the forward pass, weight values are scaled according to the number of incoming connections `n_inputs`. For layers followed by a ReLU activation function, we use He (Kaiming) Initialization.

The ReLU activation function turns all negative numbers into zero. Because ReLU silences roughly half of all incoming values, it reduces the overall signal strength by half. To balance this drop and keep activation strength consistent across layers, the variance of the initial weights must be multiplied by two.

To implement He initialization, weights are drawn from a standard normal distribution with a mean of zero and scaled by multiplying with the square root of two divided by `n_inputs`:

$$W = \mathcal{N}(0, 1) \cdot \sqrt{\frac{2.0}{n_{\text{inputs}}}}$$

Bias parameters do not scale input features, so initializing them as zero vectors keeps the initial calculations centered and balanced.

### Code Implementation in Linear Layer

In our project codebase, weight generation, scaling, and parameter setup are handled directly inside the constructor of the `Linear` class.

```python
class Linear:
    def __init__(self, n_inputs, n_neurons):
        # He/Kaiming initialization
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros(n_neurons)
        self.saved_input = None
