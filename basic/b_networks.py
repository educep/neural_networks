"""
Created by Analitika at 07/02/2025
contact@analitika.fr
"""

# External imports
import numpy as np
from numpy.typing import NDArray
from numpy import float64
from typing import List
import math
import matplotlib.pyplot as plt

# Internal imports


def gradient_step(
    v: NDArray[float64], gradient: NDArray[float64], step_size: float
) -> NDArray[float64]:
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert v.shape == gradient.shape
    return v + step_size * gradient


# ACTIVATION FUNCTIONS
def step_function(x: float) -> float:
    """Step function activation."""
    return 1.0 if x >= 0 else 0.0


def sigmoid(t: float64) -> float64:
    """Sigmoid activation function."""
    return 1 / (1 + math.exp(-t))


def perceptron(weights: NDArray[float64], bias: float, x: NDArray[float64]) -> float:
    """Perceptron model."""
    return step_function(np.dot(weights, x) + bias)


def neuron(weights: NDArray[float64], inputs: NDArray[float64]) -> float64:
    """Single neuron computation with sigmoid activation."""
    return sigmoid(np.dot(weights, inputs))


def feed_forward(
    neural_network: List[NDArray[float64]], input_vector: NDArray[float64]
) -> List[NDArray[float64]]:
    """Feeds the input vector through the neural network."""
    outputs = []
    current_input = np.append(input_vector, 1.0)

    for layer in neural_network:
        output = np.array([neuron(weights, current_input) for weights in layer])
        outputs.append(output)
        current_input = np.append(output, 1.0)  # Add bias for next layer

    return outputs


def check_xor():
    xor_network = [  # hidden layer
        np.array([[20.0, 20, -30], [20.0, 20, -10]]),  # 'and' neuron  # 'or'  neuron
        # output layer
        np.array([[-60.0, 60, -30]]),
    ]  # '2nd input but not 1st input' neuron

    # feed_forward returns the outputs of all layers, so the [-1] gets the
    # final output, and the [0] gets the value out of the resulting vector
    assert 0.000 < feed_forward(xor_network, [0, 0])[-1][0] < 0.001
    assert 0.999 < feed_forward(xor_network, [1, 0])[-1][0] < 1.000
    assert 0.999 < feed_forward(xor_network, [0, 1])[-1][0] < 1.000
    assert 0.000 < feed_forward(xor_network, [1, 1])[-1][0] < 0.001


def sqerror_gradients(
    network: List[NDArray[float64]],
    input_vector: NDArray[float64],
    target_vector: NDArray[float64],
) -> List[NDArray[float64]]:
    """Computes the gradient of the squared error loss with respect to the neuron weights."""
    layer_outputs = feed_forward(network, input_vector)
    outputs = layer_outputs[-1]
    hidden_outputs = np.append(
        layer_outputs[0], 1.0
    )  # Ensure consistency with bias handling

    output_deltas = outputs * (1 - outputs) * (outputs - target_vector)
    output_grads = np.outer(output_deltas, hidden_outputs)

    hidden_weights = network[-1]  # Access last layer's weights
    hidden_weights_no_bias = np.array(
        [neuron_weights[:-1] for neuron_weights in hidden_weights]
    )  # Remove bias

    hidden_deltas = (
        hidden_outputs[:-1]
        * (1 - hidden_outputs[:-1])
        * np.dot(hidden_weights_no_bias.T, output_deltas)
    )
    hidden_grads = np.outer(hidden_deltas, np.append(input_vector, 1.0))

    return [hidden_grads, output_grads]


def main():
    import random
    import tqdm

    random.seed(0)

    xs = np.array([binary_encode(n) for n in range(101, 1024)])
    ys = np.array([fizz_buzz_encode(n) for n in range(101, 1024)])

    NUM_HIDDEN = 25
    network = [
        # np.random.rand(NUM_HIDDEN, 10 + 1),
        np.array([[random.random() for _ in range(10 + 1)] for _ in range(NUM_HIDDEN)]),
        # np.random.rand(4, NUM_HIDDEN + 1)
        np.array([[random.random() for _ in range(NUM_HIDDEN + 1)] for _ in range(4)]),
    ]

    learning_rate = 1.0
    with tqdm.trange(200) as t:
        for epoch in t:
            epoch_loss = 0.0
            for x, y in zip(xs, ys):
                predicted = feed_forward(network, x)[-1]
                epoch_loss += np.sum((predicted - y) ** 2)
                gradients = sqerror_gradients(network, x, y)

                network = np.array(
                    [
                        [
                            gradient_step(neuron, grad, -learning_rate)
                            for neuron, grad in zip(layer, layer_grad)
                        ]
                        for layer, layer_grad in zip(network, gradients)
                    ],
                    dtype=object,
                )

            t.set_description(f"fizz buzz (loss: {epoch_loss:.2f})")

    num_correct = sum(
        np.argmax(feed_forward(network, binary_encode(n))[-1])
        == np.argmax(fizz_buzz_encode(n))
        for n in range(1, 101)
    )
    print(f"{num_correct} / 100")


def binary_encode(x: int) -> NDArray[np.float64]:
    binary = []

    for i in range(10):
        binary.append(x % 2)
        x = x // 2

    return np.array(binary, dtype=np.float64)


def fizz_buzz_encode(x: int) -> NDArray[np.float64]:
    if x % 15 == 0:
        outp = [0, 0, 0, 1]
    elif x % 5 == 0:
        outp = [0, 0, 1, 0]
    elif x % 3 == 0:
        outp = [0, 1, 0, 0]
    else:
        outp = [1, 0, 0, 0]

    return np.array(outp, dtype=np.float64)


if __name__ == "__main__":
    main()
