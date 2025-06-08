# Multilayer Perceptron (MLP) Implementation for XOR and Iris Classification

This project implements a multilayer perceptron (MLP) from scratch using NumPy.  It demonstrates the MLP's ability to learn the XOR problem and perform Iris flower classification.  The project also explores the effects of varying hyperparameters like the number of hidden neurons and the learning rate.

## Project Overview

This Jupyter Notebook (`XOR_MLP v1.ipynb`) contains:

1. **MLP Implementation:** A general-purpose `MLP` class is defined, capable of handling various numbers of input, hidden, and output neurons.  The implementation includes:
    - Sigmoid activation function and its derivative.
    - Feedforward propagation.
    - Backpropagation algorithm for training.
    - Gradient descent for weight and bias updates.
    - A `predict` method for making classifications after training.

2. **XOR Problem:** The MLP is trained on the XOR dataset to demonstrate its ability to solve a non-linearly separable problem.  The impact of altering the number of hidden neurons (2 vs 3) on training speed is analyzed graphically.

3. **Iris Classification:**  The MLP is used to classify the Iris flower dataset. The dataset is preprocessed to convert categorical features into numerical representations suitable for the MLP. The performance is evaluated and results are presented.

4. **Wine Dataset Classification (Extra):**  An additional experiment is conducted using the Wine dataset. This demonstrates scaling and preprocessing techniques and tests the robustness of the implemented MLP.

## Results and Analysis

- **XOR Problem:** The 3-neuron hidden layer MLP converges faster than the 2-neuron version, showcasing the potential benefits of increased model capacity.  Graphs illustrate the training cost across epochs for both models, highlighting the faster convergence with more neurons.

- **Iris Dataset:** The MLP successfully classifies the Iris dataset with high accuracy, demonstrating the algorithm's ability to learn complex patterns.

- **Wine Dataset:** The Wine dataset shows excellent results after data standardization (using `StandardScaler`). This reinforces the importance of proper data preprocessing.

## Datasets

The project uses three datasets:

- **XOR dataset:** A simple, built-in dataset for demonstrating the MLP's capabilities.
- **Iris dataset:** The data is included in the repository as `iris_data.csv`.
- **Wine dataset:** The data is included in the repository as `wine_data.csv`.

## Future Work

- Implement different activation functions (e.g., ReLU, tanh).
- Explore different optimization algorithms (e.g., Adam, RMSprop).
- Add more sophisticated regularization techniques (e.g., dropout, weight decay).
- Compare the performance of the MLP against other classification algorithms.
- Conduct a more thorough hyperparameter search
