"""
Created by Analitika at 07/02/2025
contact@analitika.fr
"""

# External imports
import numpy as np
from basic.a_neurons import perceptron


def define_boolean_operator_and():
    """
    we can create an AND gate, which returns 1 if both its
    inputs are 1 but returns 0 if one of its inputs is 0
    """ ""
    and_weights = np.array([2.0, 2.0])
    and_bias = -3.0
    assert perceptron(and_weights, and_bias, np.array([1, 1])) == 1
    assert perceptron(and_weights, and_bias, np.array([0, 1])) == 0
    assert perceptron(and_weights, and_bias, np.array([1, 0])) == 0
    assert perceptron(and_weights, and_bias, np.array([0, 0])) == 0


def define_boolean_operator_or():
    """
    we can create an OR gate, which returns 1 if at least one of its inputs is 1
    but returns 0 if both of its inputs are 0
    """
    or_weights = np.array([2.0, 2.0])
    or_bias = -1.0
    assert perceptron(or_weights, or_bias, np.array([1, 1])) == 1
    assert perceptron(or_weights, or_bias, np.array([0, 1])) == 1
    assert perceptron(or_weights, or_bias, np.array([1, 0])) == 1
    assert perceptron(or_weights, or_bias, np.array([0, 0])) == 0


def define_boolean_operator_not():
    """
    we can create a NOT gate, which returns 1 when its input is 0 and
    returns 0 when its input is 1
    """
    not_weights = np.array([-2.0])
    not_bias = 1.0
    assert perceptron(not_weights, not_bias, np.array([0])) == 1
    assert perceptron(not_weights, not_bias, np.array([1])) == 0


def main():
    define_boolean_operator_and()
    define_boolean_operator_or()
    define_boolean_operator_not()

    """
    Of course, you don't need to approximate a neuron in order to build a logic gate:
    and_gate = min
    or_gate = max
    xor_gate = lambda x, y: 0 if x == y else 1
    """
