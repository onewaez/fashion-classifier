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
```
## Stage 3: ReLU Activation Function

### The Fundamental Role of Non-Linearity

A neural network composed exclusively of linear layers is mathematically limited in its expressive capacity. When multiple linear transformations are stacked sequentially, their combined operation simplifies into a single affine transformation. Multiplying a linear output $Z_1 = X \cdot W_1 + b_1$ by a second weight matrix $W_2$ produces $Z_2 = X \cdot (W_1 \cdot W_2) + (b_1 \cdot W_2 + b_2)$, which is functionally equivalent to a single linear layer with merged parameters $W_{\text{new}}$ and $b_{\text{new}}$. Without intermediate non-linear operations, a multi-layer architecture collapses into a simple linear regression model incapable of learning complex visual boundaries.

To separate 10 distinct clothing categories in Fashion-MNIST, the network must fold and warp high-dimensional feature space. Non-linear activation functions introduce non-linear decision boundaries between layers, enabling hidden neurons to combine low-level visual features into complex geometric representations.

### Mechanics of the Rectified Linear Unit

Our architecture implements the Rectified Linear Unit (ReLU) activation function after the hidden dense layer. ReLU applies a threshold operation that replaces all negative input values with zero while allowing positive values to pass through unchanged:

$$f(x) = \max(0, x)$$

Compared to traditional smooth activation functions such as Sigmoid or Tanh, ReLU offers two major mathematical advantages. First, it requires negligible computational overhead, replacing costly exponential calculations with a simple threshold operation. Second, ReLU mitigates the vanishing gradient problem. Activation functions like Sigmoid saturate at extreme values, causing their derivatives to approach zero and halting parameter updates in deep layers. In contrast, the derivative of ReLU for any positive input is exactly 1.0, allowing gradient signals to pass backward without exponential decay.

### Backward Pass and Gradient Masking

During backpropagation, error gradients flow backward from the loss function toward earlier layers. To calculate how much a pre-activation value influenced the final loss, the chain rule requires multiplying the incoming upstream gradient by the local derivative of the activation function.

The mathematical derivative of the ReLU function is step-based:

$$f'(x) = \begin{cases} 1 & \text{if } x > 0, \\ 0 & \text{if } x \le 0 \end{cases}$$

When an input value was positive during the forward pass ($x > 0$), ReLU acts as an open channel, passing the upstream gradient backward without modification. When an input value was zero or negative ($x \le 0$), the derivative is 0, creating a binary gradient mask that completely blocks upstream error signals from passing backward through that specific neuron.

<img width="1920" height="1080" alt="Stage3ReLUDiagram_ManimCE_v0 22 0" src="https://github.com/user-attachments/assets/8f840889-fecb-4298-b743-0e67f4d655f0" />


### Python Implementation

In our codebase, the `ReLU` class manages non-linear transformations and gradient filtering. The forward pass stores a reference to the unactivated input tensor $X$, which is reused during backpropagation to generate a boolean mask for element-wise gradient multiplication.

```python
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
```
## Stage 4: Forward Pass & Categorical Cross-Entropy Loss

### Sequential Feedforward Execution

The complete forward pass integrates individual network layers into a unified computing pipeline that transforms raw input pixels into predicted class probabilities. Given a mini-batch of normalized image vectors $X$ of shape `(N, 784)`, the data flows sequentially through the hidden layer, non-linear activation, and output layer.

First, the input batch passes through `Linear(784, 128)`, producing pre-activation tensor $Z_1 = X \cdot W_1 + b_1$ of shape `(N, 128)`. Next, the `ReLU` activation function filters negative activations to yield hidden feature tensor $A_1 = \max(0, Z_1)$. Finally, the hidden activations pass through `Linear(128, 10)` to compute the raw unnormalized output tensor $Z_2 = A_1 \cdot W_2 + b_2$ of shape `(N, 10)`. These final scalar outputs, called logits, represent uncalibrated class scores for each of the 10 clothing categories.

### Softmax Probabilities and Numerical Stability

Because raw logits can be arbitrary real numbers ranging from negative to positive infinity, they cannot be directly interpreted as class probabilities. To convert logits into a valid probability distribution where all values are positive and sum to 1.0 across all 10 classes, the network applies the Softmax activation function.

Mathematically, the Softmax function exponentiates each raw logit $z_i$ and divides it by the sum of exponentiated logits across all classes:

$$P_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Directly computing exponentials of large positive logits introduces severe numerical instability, triggering floating-point overflow errors in NumPy. To guarantee numerical stability, the maximum logit value within each sample row is subtracted from all logits prior to exponentiation. Subtracting a constant shift factor $C = \max(z)$ scales the maximum exponent to $e^0 = 1.0$ without altering the resulting probability ratios:

$$P_i = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{K} e^{z_j - \max(z)}}$$

### Categorical Cross-Entropy Loss Formulation

To evaluate model performance during training, predicted probability distributions are compared against target labels using the Categorical Cross-Entropy loss function. Cross-entropy quantifies the dissimilarity between the predicted probability distribution $P$ and the true target distribution $Y$.

For a single training sample with target class index $y$, the loss penalizes low probability assignments to the correct target label using the negative natural logarithm:

$$L = -\ln(P_y)$$

When the network assigns a high probability close to 1.0 to the correct class, the loss approaches zero. Conversely, if the network assigns a low probability near zero to the correct class, the negative logarithm expands exponentially toward infinity, imposing a heavy penalty. Over a mini-batch of $N$ samples, the total loss is computed as the average loss across all individual examples:

$$\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \ln(P_{i, y_i})$$

<img width="1920" height="1080" alt="Stage4LossDiagram_ManimCE_v0 22 0" src="https://github.com/user-attachments/assets/1744ea12-4c74-4841-81d6-38f7c87c270a" />


### Code Implementation

In our modular codebase, Softmax activation and Categorical Cross-Entropy loss are combined into the `CrossEntropyLoss` class. The implementation handles numerical shift subtraction, probability computation, log stabilization, and mini-batch loss averaging.

```python
class CrossEntropyLoss:
    def __init__(self):
        self.saved_probabilities = None

    def forward(self, logits, correct_class_indices):
        # Softmax with max subtraction for numerical stability
        max_logits = np.max(logits, axis=1, keepdims=True)
        safe_logits = logits - max_logits
        exponents = np.exp(safe_logits)
        
        self.saved_probabilities = exponents / np.sum(exponents, axis=1, keepdims=True)
        
        batch_size = logits.shape[0]
        correct_probs = self.saved_probabilities[np.arange(batch_size), correct_class_indices]
        
        log_probs = np.log(correct_probs + 1e-9)
        loss = -np.sum(log_probs) / batch_size
        
        return loss

    def backward(self, logits, correct_class_indices):
        batch_size = logits.shape[0]
        start_gradient = self.saved_probabilities.copy()
        start_gradient[np.arange(batch_size), correct_class_indices] -= 1.0
        
        return start_gradient / batch_size
```

## Stage 5: Backpropagation & Stochastic Gradient Descent (SGD) — Part 1

### Mathematical Foundations of Backpropagation

Backpropagation constitutes the primary optimization mechanism of a neural network, translating overall classification error into precise parameter adjustments across all hidden and output transformations. While the forward pass processes raw input pixel arrays through successive linear projections and activation functions to compute class probabilities, the backward pass reverses this direction. By systematically evaluating the partial derivatives of the scalar loss function with respect to every weight matrix and bias vector, the network determines the exact contribution of each parameter to prediction error.

Error propagation relies on the multivariable calculus chain rule, which decomposes complex nested function derivatives into a sequence of localized matrix multiplications. Signals flow backward from the loss layer through output weights, activation filters, and hidden projections down to the input features. Each module receives an incoming gradient tensor representing the error from downstream layers, computes local partial derivatives to update its internal weights and biases, and returns an upstream gradient tensor to preceding operations.

### Step 1: Initial Loss Gradient Computation

The backward pass originates inside the loss function by evaluating the derivative of the Categorical Cross-Entropy loss with respect to the raw, unnormalized logit outputs produced by the final linear transformation. Mathematically combining the softmax probability mapping with cross-entropy loss yields an analytical derivative that simplifies to the normalized difference between predicted probabilities and ground-truth target values:

$$\frac{\partial L}{\partial Z_2} = \frac{1}{N} (P - Y)$$

In this formulation, $P$ represents the matrix of predicted class probabilities of shape `(N, 10)` computed during the forward pass, $Y$ denotes the one-hot encoded matrix of true target labels of shape `(N, 10)`, and $N$ represents the mini-batch size `batch_size`. For every non-target class, the gradient component equals the predicted probability scaled by batch size, whereas for the correct target class, unity is subtracted from the predicted probability before scaling.

```python
def backward(self, logits, correct_class_indices):
    batch_size = logits.shape[0]
    start_gradient = self.saved_probabilities.copy()
    start_gradient[np.arange(batch_size), correct_class_indices] -= 1.0
    
    return start_gradient / batch_size
```

The backward method of CrossEntropyLoss executes this calculation by extracting batch_size directly from logits.shape[0] and creating an independent duplicate of cached probability values stored in self.saved_probabilities. Array indexing via np.arange(batch_size) paired with correct_class_indices isolates target probabilities across all batch rows and subtracts 1.0 in place. Dividing the updated array by batch_size normalizes error magnitudes across mini-batch samples, yielding an initial error tensor start_gradient of shape (N, 10) ready for backward propagation.

## Stage 5: Backpropagation & Stochastic Gradient Descent (SGD) — Step 2: Output Layer Matrix Calculus and Parameter Updates

The error gradient tensor calculated by the loss function enters the second transformation layer, `Linear(128, 10)`. The backward execution phase inside a fully connected linear layer evaluates three interconnected matrix derivative operations: computing partial derivatives with respect to internal weight matrix $W_2$, computing gradients with respect to bias vector $b_2$, and deriving the activation gradient tensor propagated upstream to hidden layer activations.

### Weight Gradient Derivation

The partial derivative of scalar loss function $L$ with respect to weight matrix $W_2$ measures how individual weight parameters influence output classification errors across all mini-batch samples. Mathematically, this derivative is derived by multiplying transposed hidden activation matrix $A_1^T$ of shape `(128, N)` saved during forward execution by incoming logit error matrix $dZ_2$ of shape `(N, 10)`:

$$\frac{\partial L}{\partial W_2} = A_1^T \cdot dZ_2$$

Evaluating this matrix dot product accumulates individual error contributions across $N$ mini-batch samples. The resulting weight gradient matrix $dW_2$ shares the exact structural dimensions `(128, 10)` of weight matrix $W_2$, where each element $(j, k)$ quantifies the sensitivity of the loss function to changes in the connection weight between hidden neuron $j$ and output class $k$.

### Bias Gradient Accumulation

The bias vector shifts class logit scores independently of hidden feature activation intensities. The gradient of the loss function with respect to bias vector $b_2$ is evaluated by summing logit error vectors across all mini-batch samples along the sample dimension:

$$\frac{\partial L}{\partial b_2} = \sum_{i=1}^{N} dZ_{2, i}$$

Summing error components across mini-batch rows collapses the sample dimension $N$, producing a one-dimensional bias gradient array $db_2$ of shape `(10,)` that matches the structural dimensions of bias vector $b_2$.

### Upstream Activation Gradient Propagation

To propagate error information backward through the architecture, the incoming logit gradient matrix $dZ_2$ must be projected back into the hidden activation space. Multiplying incoming error matrix $dZ_2$ of shape `(N, 10)` by transposed weight matrix $W_2^T$ of shape `(10, 128)` projects classification errors back through the layer parameters:

$$dA_1 = dZ_2 \cdot W_2^T$$

The resulting activation error tensor $dA_1$ has dimensions `(N, 128)`, exactly matching the shape of hidden layer activations $A_1$ produced during forward evaluation. This upstream error matrix quantifies how much each hidden activation unit contributed to output logit errors, preparing the error signal for activation filtering in preceding layers.

### Parameter Optimization via Stochastic Gradient Descent

Once parameter gradients $dW_2$ and $db_2$ are derived, Stochastic Gradient Descent adjusts internal weight and bias values in place. Parameters are updated along the negative gradient vector direction, scaled by a hyperparameter learning rate $\alpha$:

$$W_2 \leftarrow W_2 - \alpha \cdot dW_2$$

$$b_2 \leftarrow b_2 - \alpha \cdot db_2$$

Subtracting scaled gradient matrices systematically reduces overall cross-entropy loss by adjusting parameters in the direction that decreases classification error, concluding optimization for the output layer before returning upstream activation gradient tensor $dA_1$.

### Stage 5: Backpropagation & SGD — Step 3 (Activation Gradient Filtering via ReLU)

#### Theoretical Overview & Chain Rule Formulation

Following the computation of parameter gradients for the output layer, the backward error propagation proceeds to the hidden activation layer. During the forward pass, the Rectified Linear Unit (ReLU) activation function transformed the output of the first linear layer—the pre-activation matrix $Z_1 \in \mathbb{R}^{N \times 128}$—into activated hidden representations $A_1 = \max(0, Z_1) \in \mathbb{R}^{N \times 128}$ by applying an element-wise threshold. To propagate the loss derivative back toward the input layer, we must compute the gradient of the scalar loss function $\mathcal{L}$ with respect to the hidden pre-activations, denoted as $dZ_1 = \frac{\partial \mathcal{L}}{\partial Z_1} \in \mathbb{R}^{N \times 128}$.

By applying the multivariable chain rule, the error gradient $dZ_1$ is computed as the element-wise Hadamard product (denoted by $\odot$) of the incoming gradient from the subsequent layer, $dA_1 = \frac{\partial \mathcal{L}}{\partial A_1} \in \mathbb{R}^{N \times 128}$, and the local derivative of the ReLU function $f'(Z_1)$:

$$dZ_1 = \frac{\partial \mathcal{L}}{\partial Z_1} = \frac{\partial \mathcal{L}}{\partial A_1} \odot f'(Z_1)$$

Because the ReLU function $f(z) = \max(0, z)$ is piecewise linear, its element-wise derivative with respect to input $z$ is defined as:

$$f'(z) = \begin{cases} 1 & \text{if } z > 0, \\ 0 & \text{if } z \le 0 \end{cases}$$

Substituting this local derivative back into the chain rule formulation yields the explicit mathematical rule for backpropagation through a ReLU activation layer:

$$dZ_1 = \frac{\partial \mathcal{L}}{\partial Z_1} = \frac{\partial \mathcal{L}}{\partial A_1} \odot \mathbb{I}(Z_1 > 0)$$

where $\mathbb{I}(Z_1 > 0)$ denotes an element-wise binary indicator function that evaluates to $1$ for coordinates where $Z_1 > 0$ and $0$ where $Z_1 \le 0$.

---

#### Intuitive Mechanics: The Dynamic Gradient Filter

To understand this process intuitively, the derivative of ReLU acts as a network of dynamic binary switches—or an ON/OFF filter—operating on every feature for every sample in a mini-batch:

1. **Active Neurons ($Z_1 > 0$):** When a pre-activation unit was strictly positive during the forward pass, its output changed linearly with its input with a constant slope of $1$. As a result, its derivative $f'(z)$ is $1$. Multiplying the incoming gradient $dA_1$ by $1$ leaves the gradient completely unchanged in both magnitude and direction, allowing the error signal to flow backward unimpeded.
2. **Inactive Neurons ($Z_1 \le 0$):** When a pre-activation unit was zero or negative during the forward pass, its output was clamped to $0$. Because small changes in negative inputs produce zero change in the activated output, the local slope is $0$. Multiplying the incoming gradient $dA_1$ by $0$ blocks the error signal from traveling further backward through that neuron.

This selective gating mechanism ensures that parameters are updated exclusively through pathways that actively contributed to the forward prediction. Inactive units receive a zero gradient, preventing unnecessary adjustments to weights that played no role in generating the model's output for that forward step.

---

#### Vectorized Implementation & Memory Management

In our custom framework, this filtering step is handled within the `backward` method of the `ReLU` class inside `layers.py`.

During the forward pass, the layer caches its input tensor in `self.saved_input`. When `Sequential.backward` executes during backpropagation, it invokes `ReLU.backward` with two arguments:
- `gradient_from_next_layer`: The incoming gradient matrix $dA_1 \in \mathbb{R}^{N \times 128}$.
- `learning_rate`: The scalar optimization step size.

Because activation functions contain no learnable parameters (weights or biases), the `learning_rate` parameter is unused inside the method body. It is retained strictly to maintain a uniform method interface across all layer objects inside the `Sequential` execution container.

The method performs two vectorized operations:
1. **Memory Isolation:** To prevent mutating the original `gradient_from_next_layer` matrix (which may be referenced elsewhere in memory), the implementation creates an independent array copy: `gradient_for_previous_layer = np.array(gradient_from_next_layer, copy=True)`.
2. **Vectorized Masking:** Using NumPy's boolean indexing, `self.saved_input <= 0` constructs an $N \times 128$ boolean mask identifying all non-positive entries. Executing `gradient_for_previous_layer[self.saved_input <= 0] = 0` zeroes out all blocked gradient paths across the entire batch in a single optimized C-level CPU operation.

The returned matrix `gradient_for_previous_layer` represents $dZ_1 \in \mathbb{R}^{N \times 128}$ and serves directly as the incoming gradient tensor for the initial `Linear(784, 128)` layer.

---

#### Code Implementation

```python
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
```
### Stage 5: Backpropagation & SGD — Step 4 (First Layer Parameter Updates for `Linear(784, 128)`)

#### Theoretical Overview & Matrix Calculus

Having filtered the error gradient through the ReLU activation function, backpropagation reaches the first fully connected layer, `Linear(784, 128)`. At this point, the layer receives the incoming gradient matrix $dZ_1 = \frac{\partial \mathcal{L}}{\partial Z_1} \in \mathbb{R}^{N \times 128}$ from the activation step. This layer contains the network's first set of trainable parameters: the weight matrix $W_1 \in \mathbb{R}^{784 \times 128}$ and the bias vector $b_1 \in \mathbb{R}^{128}$. To update these parameters, we must compute the exact partial derivatives of the loss function $\mathcal{L}$ with respect to $W_1$ and $b_1$.

During the forward pass, the linear transformation was defined as:

$$Z_1 = X \cdot W_1 + b_1$$

where $X \in \mathbb{R}^{N \times 784}$ represents the input batch of flattened $28 \times 28$ Fashion-MNIST images cached as `self.saved_input`. Applying matrix calculus and the multivariable chain rule allows us to derive three distinct gradients:

1. **Gradient with Respect to Weights ($\frac{\partial \mathcal{L}}{\partial W_1}$):**
   The sensitivity of the loss to changes in each weight depends on the input feature values $X$ and the error signal $dZ_1$. Transposing the input matrix $X^T \in \mathbb{R}^{784 \times N}$ aligns the mini-batch dimension $N$ for matrix multiplication:

   $$\frac{\partial \mathcal{L}}{\partial W_1} = X^T \cdot dZ_1 \quad \in \mathbb{R}^{784 \times 128}$$

2. **Gradient with Respect to Biases ($\frac{\partial \mathcal{L}}{\partial b_1}$):**
   Because biases are added identically to every sample in the mini-batch, the total gradient for each bias scalar is the sum of error signals across all $N$ samples in the batch:

   $$\frac{\partial \mathcal{L}}{\partial b_1} = \sum_{i=1}^{N} dZ_{1, i} \quad \in \mathbb{R}^{128}$$

3. **Gradient with Respect to Input Data ($\frac{\partial \mathcal{L}}{\partial X}$):**
   To allow backward error flow across deeper or stacked network architectures, the layer computes the gradient with respect to its input $X$ by projecting $dZ_1$ back through the transposed weight matrix $W_1^T \in \mathbb{R}^{128 \times 784}$:

   $$\frac{\partial \mathcal{L}}{\partial X} = dZ_1 \cdot W_1^T \quad \in \mathbb{R}^{N \times 784}$$

---

#### Intuitive Mechanics: Updating Input Feature Detectors

In an MLP, the weights of the first layer $W_1$ act as visual feature detectors. Each of the $128$ hidden neurons connects to all $784$ pixel locations of an input image, searching for foundational visual patterns such as edges, curves, or intensity transitions characteristic of specific clothing categories (e.g., sleeves, shoe soles, or collar lines).

- **Weight Updates ($\frac{\partial \mathcal{L}}{\partial W_1}$):** The gradient entry $\frac{\partial \mathcal{L}}{\partial W_{1, (j, k)}}$ measures how much the loss would increase or decrease if the connection weight between pixel $j$ and hidden unit $k$ were altered. The dot product $X^T \cdot dZ_1$ naturally scales the gradient by the pixel brightness $X$: if a pixel was dark ($0.0$), it contributed nothing to the neuron's activation, so its weight receives zero gradient. Conversely, bright pixels that contributed strongly to an incorrect prediction receive significant gradient adjustments.
- **Bias Updates ($\frac{\partial \mathcal{L}}{\partial b_1}$):** The bias gradient measures the overall baseline activation tendency for each hidden unit across the entire mini-batch, independent of specific input pixel configurations.

---

#### Vectorized Implementation & SGD Optimization

In our framework, parameter gradients and update operations are executed inside the `backward` method of the `Linear` class in `layers.py`.

The method takes `gradient_from_next_layer` ($dZ_1$) and `learning_rate` ($\eta$) as arguments, carrying out the calculations through efficient NumPy linear algebra operations:

1. **Input Gradient Calculation:**
   `gradient_for_previous_layer = np.dot(gradient_from_next_layer, self.weights.T)` computes $dX \in \mathbb{R}^{N \times 784}$. This tensor is returned to the caller, completing the backpropagation pipeline.
2. **Weight Gradient Calculation:**
   `gradients_for_weights = np.dot(self.saved_input.T, gradient_from_next_layer)` computes $\frac{\partial \mathcal{L}}{\partial W_1} \in \mathbb{R}^{784 \times 128}$.
3. **Bias Gradient Calculation:**
   `gradients_for_biases = np.sum(gradient_from_next_layer, axis=0)` sums errors across batch samples along `axis=0`, yielding $\frac{\partial \mathcal{L}}{\partial b_1} \in \mathbb{R}^{128}$.
4. **Stochastic Gradient Descent (SGD) Step:**
   Immediately following gradient computation, Stochastic Gradient Descent modifies the layer's parameters in-place, stepping in the opposite direction of the gradient to minimize the loss:

   $$W_1 \leftarrow W_1 - \eta \cdot \frac{\partial \mathcal{L}}{\partial W_1}$$

   $$b_1 \leftarrow b_1 - \eta \cdot \frac{\partial \mathcal{L}}{\partial b_1}$$

   In code, this is expressed as `self.weights -= learning_rate * gradients_for_weights` and `self.biases -= learning_rate * gradients_for_biases`.

---

#### Code Implementation

```python
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
```

### Stage 5: Backpropagation & SGD — Step 5 (Sequential Execution Loop and Optimization Integration)

#### Theoretical Overview & Pipeline Orchestration

Having analyzed the step-by-step mathematical operations inside each layer, the final component of the optimization framework is the global orchestration of forward and backward passes. Deep learning architectures consist of ordered computational graphs where data flows sequentially from the input space to prediction logits, while error gradients flow in the exact reverse direction. To manage this bidirectional data movement without manually wiring dependencies between individual layers, our framework uses a modular container class named `Sequential`, defined in `model.py`.

The `Sequential` model acts as an abstraction layer that holds an ordered sequence of network components—in our case, `Linear(784, 128)`, `ReLU()`, and `Linear(128, 10)`. Decoupling the orchestration logic from individual layer mechanics enforces strict modularity. Each layer standardizes its interface by providing a `forward(input_data)` method and a `backward(gradient_from_next_layer, learning_rate)` method. This design pattern allows arbitrary network topologies to be assembled and trained seamlessly, as the container only needs to pass tensors sequentially along the layer list during evaluation and in reverse during backpropagation.

---

#### Reverse Order Traversal & Gradient Cascading

The core mechanism enabling modular backpropagation is reverse sequence iteration. Because the multivariable chain rule requires calculating derivatives from the scalar loss output back toward the input features, error propagation must start at the loss function and work backward layer by layer. 

During the backward call, `Sequential.backward` receives the initial loss gradient $dZ_2 = \frac{1}{N}(P - Y) \in \mathbb{R}^{N \times 10}$ produced by `CrossEntropyLoss.backward`. The container iterates through the stored layer list in reverse order using Python's `reversed()` iterator:

1. **Output Layer (`Linear(128, 10)`):** Receives $dZ_2$, computes parameter gradients $\frac{\partial \mathcal{L}}{\partial W_2}$ and $\frac{\partial \mathcal{L}}{\partial b_2}$, updates $W_2$ and $b_2$ via SGD, and returns the input error gradient $dA_1 = dZ_2 \cdot W_2^T \in \mathbb{R}^{N \times 128}$.
2. **Hidden Activation (`ReLU()`):** Receives $dA_1$, applies the boolean mask $dA_1 \odot \mathbb{I}(Z_1 > 0)$, leaves parameters untouched, and returns the filtered pre-activation gradient $dZ_1 \in \mathbb{R}^{N \times 128}$.
3. **Input Layer (`Linear(784, 128)`):** Receives $dZ_1$, computes parameter gradients $\frac{\partial \mathcal{L}}{\partial W_1}$ and $\frac{\partial \mathcal{L}}{\partial b_1}$, updates $W_1$ and $b_1$ via SGD, and returns $dX \in \mathbb{R}^{N \times 784}$.

By assigning the return value of each layer's `backward` method to the variable tracking the current gradient tensor, the framework forms an uninterrupted gradient cascade across the entire architecture.

---

#### Mini-Batch SGD & Full Epoch Training Loop

The integration of forward inference, loss calculation, backpropagation, and SGD optimization is executed within the main training loop in `train.py`. Training runs for $100$ epochs across $60,000$ training samples using mini-batches of size $N = 32$ and a fixed learning rate of $\eta = 0.01$.

At the beginning of each epoch, the dataset is randomly shuffled using `np.random.permutation(n_samples)`. Shuffling breaks temporal correlations between consecutive training examples, ensuring that each mini-batch provides an unbiased stochastic estimate of the true gradient over the full data distribution. The dataset is then divided into mini-batches of $32$ samples.

For each mini-batch, the training loop performs four sequential steps:
1. **Forward Propagation:** `logits = model.forward(X_batch)` feeds the batch $X \in \mathbb{R}^{32 \times 784}$ through the network container to produce output pre-activations $Z_2 \in \mathbb{R}^{32 \times 10}$.
2. **Loss Evaluation:** `loss = criterion.forward(logits, y_batch)` evaluates numerical cross-entropy loss and caches prediction probabilities internally.
3. **Initial Gradient Generation:** `start_gradient = criterion.backward(logits, y_batch)` produces the initial error matrix $dZ_2 \in \mathbb{R}^{32 \times 10}$.
4. **Backward Propagation & Parameter Updates:** `model.backward(start_gradient, learning_rate)` triggers the reverse execution loop, updating all model weights and biases in-place across all layers.

---

#### Code Implementation

```python
# Sequential Container (model.py)
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


# Full Training Loop Integration (train.py)
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

epochs = 100
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
```
### Stage 6: Real-World Inference & Preprocessing Pipeline

#### Theoretical Overview & The Domain Shift Challenge

Deploying a trained neural network to evaluate arbitrary user-provided photographs introduces the challenge of domain shift. While the Fashion-MNIST training dataset consists of standardized, tightly cropped $28 \times 28$ grayscale images featuring centered light garments on pure black backgrounds, real-world photos captured by cameras or mobile devices possess vastly different characteristics. Real photos typically feature high resolutions, arbitrary aspect ratios, three-channel RGB color spaces, noisy lighting, shadows, and light backgrounds.

Because our multi-layer perceptron expects a normalized input vector $x \in \mathbb{R}^{1 \times 784}$ matching the exact statistical properties of the training distribution, raw photographs cannot be fed directly into the input layer. To bridge this domain gap, an automated inference preprocessing pipeline must convert raw input images into standardized representations through geometric cropping, spatial resampling, color-space conversion, dynamic background inversion, noise filtering, and range normalization.

---

#### Preprocessing Pipeline Stages

The inference pipeline implemented in `predict.py` utilizes the Python Imaging Library (Pillow) alongside NumPy to transform arbitrary image files through six sequential operations:

1. **Grayscale Conversion:** The image is loaded and converted from a multi-channel RGB space into a single-channel grayscale matrix using `.convert('L')`. This maps color pixel values to single-byte luminosity scalars ranging from $0$ (pure black) to $255$ (pure white).
2. **Aspect Ratio Standardization:** To prevent severe geometric distortion during scaling, the image undergoes a square center crop. The pipeline determines the smaller dimension between width and height, calculates symmetrical bounding coordinates, and crops the central square region of the image.
3. **Spatial Resampling:** The cropped square image is downsampled to the required input resolution of $28 \times 28$ pixels using bilinear interpolation, producing a $28 \times 28$ NumPy array of unsigned 8-bit integers.
4. **Automatic Background Inversion:** In the Fashion-MNIST dataset, foreground objects are light pixels on a dark background (where background pixels equal $0$). Real-world photography, however, usually features dark or colored objects against bright or white backgrounds. To standardize the background polarity, the code samples the top-left corner pixel at coordinate $(0, 0)$. If its intensity exceeds $128$ (indicating a light background), the entire image matrix is inverted by computing $255 - I$.
5. **Noise Thresholding:** Ambient shadows, camera sensor noise, and uneven lighting gradients can create non-zero background values that confuse the model. The pipeline applies a hard background threshold: any pixel with an intensity below $70$ is forcibly clamped to $0$, isolating the foreground item from background artifacts.
6. **Vector Flattening & Range Normalization:** Finally, the $28 \times 28$ matrix is flattened into a 1D vector of shape $(1, 784)$ and divided by $255.0$. This scales all pixel values from byte integers $[0, 255]$ into floating-point numbers in the range $[0.0, 1.0]$.

---

#### Post-Processing & Probability Estimation

Once preprocessed, the vector $x \in \mathbb{R}^{1 \times 784}$ is passed through the network container via `logits = model.forward(my_image)`. The model yields a raw output vector $Z_2 \in \mathbb{R}^{1 \times 10}$ representing unnormalized class logits.

To interpret these logits as human-readable probabilities, the inference script executes a numerically stable Softmax transformation:

$$P_i = \frac{e^{Z_{2, i} - \max(Z_2)}}{\sum_{j=1}^{10} e^{Z_{2, j} - \max(Z_2)}}$$

Subtracting the scalar maximum $\max(Z_2)$ prevents exponential numerical overflow while preserving exact relative probability ratios. The predicted class index is selected by taking the argmax over the probability distribution:

$$\hat{y} = \arg\max_{i} (P_i)$$

The confidence score associated with the top prediction is computed as $P_{\hat{y}} \times 100\%$, and the class index is mapped to its human-readable clothing label using a predefined category dictionary (`FASHION_CLASSES`).

---

#### Code Implementation

```python
import numpy as np
from PIL import Image
from layers import Linear, ReLU
from model import Sequential

FASHION_CLASSES = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

# Reconstruct model architecture
model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
])

# Load trained weights and biases
model.load_weights("fashion_mnist_weights.npz")
print("Weights loaded successfully!")

def process_image(image_path):
    # Step 1: Load image and convert to single-channel grayscale
    img = Image.open(image_path).convert('L')
    
    # Step 2: Center crop to square aspect ratio
    width, height = img.size
    new_dim = min(width, height)
    left = (width - new_dim) / 2
    top = (height - new_dim) / 2
    right = (width + new_dim) / 2
    bottom = (height + new_dim) / 2
    img = img.crop((left, top, right, bottom))
    
    # Step 3: Resize to standard 28x28 resolution
    img = img.resize((28, 28))
    img_array = np.array(img)
    
    # Step 4: Invert colors if background is detected as light
    if img_array[0, 0] > 128:
        img_array = 255 - img_array
        
    # Step 5: Threshold background noise
    img_array[img_array < 70] = 0
    
    # Step 6: Flatten to 1D vector and normalize to [0.0, 1.0]
    return img_array.reshape(1, 784) / 255.0

# Execute inference pipeline on a custom photograph
my_image = process_image("my_photo.jpg")
logits = model.forward(my_image)

# Numerically stable Softmax calculation
exp_logits = np.exp(logits - np.max(logits))
probabilities = exp_logits / np.sum(exp_logits)

# Extract prediction index and confidence percentage
predicted_class_index = np.argmax(probabilities)
confidence = probabilities[0][predicted_class_index] * 100

print("\nModel Prediction:")
print(f"Class: {FASHION_CLASSES[predicted_class_index]}")
print(f"Confidence: {confidence:.2f}%")
```
### Execution Guide

#### Dataset Preparation

The Fashion-MNIST dataset must be placed inside the `dataset/` directory in its uncompressed binary ubyte format. The required files are:
1. `train-images-idx3-ubyte`: Training set image data ($60,000$ samples).
2. `train-labels-idx1-ubyte`: Training set ground-truth category labels.
3. `t10k-images-idx3-ubyte`: Test set image data ($10,000$ samples).
4. `t10k-labels-idx1-ubyte`: Test set ground-truth category labels.

If the dataset files are downloaded in `.gz` compressed format from the official Zalando Fashion-MNIST repository, uncompress them into the `dataset/` directory prior to running the training script.

---

#### Model Training

To initiate training of the two-layer perceptron from scratch, execute `train.py`. The script automatically parses the training images and labels, normalizes pixel intensities to the range $[0.0, 1.0]$, initializes layer weights via He/Kaiming initialization, and executes $100$ training epochs with a mini-batch size of $32$ and a learning rate of $0.01$.

```bash
python train.py
```

During execution, average mini-batch cross-entropy loss is logged for every epoch:

```text
Starting training...
Epoch 1/100, Loss: 0.5284
Epoch 2/100, Loss: 0.4121
...
Epoch 100/100, Loss: 0.1425
Training completed. Weights saved to fashion_mnist_weights.npz
```

Upon completion, the trained parameters (weights and biases for all linear layers) are saved to `fashion_mnist_weights.npz` in the root directory.

---

#### Real-World Inference on Custom Images

To evaluate a custom photograph or image file using the trained model parameters, run `predict.py`. Place your target image file (e.g., `my_photo.jpg`) in the project directory, ensure `fashion_mnist_weights.npz` is present, and execute:

```bash
python predict.py
```

The script processes the photograph through the automated preprocessing pipeline, evaluates the processed input vector through the restored `Sequential` model, and prints the top predicted clothing category along with its Softmax confidence score:

```text
Weights loaded successfully!

Model Prediction:
Class: Ankle boot
Confidence: 94.82%
```
### Conclusion & Portfolio Takeaways

This project demonstrates the complete lifecycle of a multi-layer perceptron implemented entirely from first principles using pure Python and NumPy. By deliberately omitting high-level deep learning abstractions, every computational phase-from parsing raw ubyte binary files to matrix multiplication, non-linear activation gating, numerically stable loss computation, and backpropagation via multivariable matrix calculus-is made fully transparent and explicit.

Building a white-box neural network highlights the direct connection between mathematical abstractions and computational implementation. Rather than treating neural layers as opaque black boxes, the codebase exposes how weight matrices act as visual feature detectors, how piecewise linear activations control gradient flow through binary masking, and how stochastic gradient descent systematically adjusts thousands of parameters to minimize classification error across mini-batches.

A central insight gained from this project is the critical role of tensor vectorization in scientific computing. By organizing batch inputs, weight matrices, and error gradients into multi-dimensional NumPy arrays, the entire training pipeline executes through highly optimized C-level matrix operations. This vectorized approach eliminates performance bottlenecks associated with scalar Python loops, demonstrating how mathematical formulations translate directly into hardware-efficient matrix algebra.

Furthermore, the implementation bridges the gap between theoretical models and real-world application through a robust inference preprocessing pipeline. Addressing the domain shift between clean benchmark datasets and arbitrary user photography illustrates that model architecture is only one component of a production system. Ensuring that real-world inputs match the spatial, luminosity, and statistical characteristics of the training data is essential for reliable prediction performance.

Ultimately, while frameworks such as PyTorch and TensorFlow remain indispensable for scaling complex architectures with automatic differentiation and GPU acceleration, building a deep learning framework from scratch provides foundational engineering intuition. Understanding the underlying linear algebra, calculus, and optimization mechanics equips developers to better debug training instabilities, design custom layer operations, and make informed architectural choices in advanced machine learning systems.
