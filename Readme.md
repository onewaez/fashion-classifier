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
