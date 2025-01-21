import numpy as np


def sigmoid(x):
    """
    Computes the sigmoid activation function in a numerically stable way.
    """
    positive_mask = (x >= 0)
    negative_mask = ~positive_mask
    result = np.zeros_like(x, dtype=np.float64)

    result[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))
    exp_x = np.exp(x[negative_mask])
    result[negative_mask] = exp_x / (1 + exp_x)

    return result


def sigmoid_derivative(sigmoid_output):
    """
    Computes the derivative of the sigmoid function given its output.
    """
    return sigmoid_output * (1 - sigmoid_output)


# Function to normalize data
def normalize_data(data):
    """
    Normalize the data so that each feature is between 0 and 1.
    Handles edge cases when max and min values are the same.
    """
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    range_vals = max_vals - min_vals

    # Prevent division by zero
    range_vals[range_vals == 0] = 1

    return (data - min_vals) / range_vals


# ReLU activation function
def relu(x):
    """
    ReLU activation function.
    """
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation function to avoid the "dying ReLU" problem.
    """
    return np.where(x > 0, x, alpha * x)


def relu_derivative(x):
    """
    Derivative of the ReLU function.
    """
    return np.where(x > 0, 1, 0)


def leaky_relu_derivative(x, alpha=0.01):
    """
    Derivative of the Leaky ReLU function.
    """
    return np.where(x > 0, 1, alpha)


# Softmax activation function (for classification)
def softmax(x):
    """
    Softmax activation function for multi-class classification.
    Converts outputs into probabilities.
    """
    exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Stability improvement
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


# Loss Function (Mean Squared Error)
def mean_squared_error(y_true, y_pred):
    """
    Calculates Mean Squared Error (MSE) loss.
    """
    return np.mean((y_true - y_pred) ** 2)


# Loss Function (Cross-Entropy Loss)
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-10  # Avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
