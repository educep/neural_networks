"""
Created by Eduardo Cepeda at 26/04/2024
eduardo@cepeda.fr

This module provides basic implementations of neural network components,
including activation functions and simple neuron models.
"""

# External imports
import numpy as np
from numpy.typing import NDArray
from numpy import float64
import math


# ACTIVATION FUNCTIONS
def step_function(x: float) -> float:
    """
    Step function activation.

    Args:
        x (float): Input value.

    Returns:
        float: 1.0 if x >= 0, else 0.0.
    """
    return 1.0 if x >= 0 else 0.0


def sigmoid(t: float) -> float:
    """
    Sigmoid activation function.

    Args:
        t (float): Input value.

    Returns:
        float: Sigmoid of t, ranging between 0 and 1.
    """
    return 1 / (1 + math.exp(-t))


def perceptron(weights: NDArray[float64], bias: float, x: NDArray[float64]) -> float:
    """
    Perceptron model.

    The perceptron is one of the simplest neural networks, approximating a single neuron
    with n binary inputs. It computes a weighted sum of its inputs and "fires" if the
    weighted sum is 0 or greater.

    Args:
        weights (NDArray[float64]): Weight vector for the inputs.
        bias (float): Bias term.
        x (NDArray[float64]): Input vector.

    Returns:
        float: 1 if the perceptron 'fires', 0 otherwise.
    """
    calculation = np.dot(weights, x) + bias  # Weighted sum + bias
    return step_function(calculation)


def neuron(weights: NDArray[float64], inputs: NDArray[float64]) -> float:
    """
    Simple neuron model with sigmoid activation.

    Args:
        weights (NDArray[float64]): Weight vector, including the bias term.
        inputs (NDArray[float64]): Input vector, including a 1 for the bias term.

    Returns:
        float: Output of the neuron after applying the sigmoid function.
    """
    return sigmoid(np.dot(weights, inputs))  # Weighted sum + sigmoid activation
