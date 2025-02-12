"""
Created by Analitika at 09/02/2025
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

biais = 1.0


def gradient_step(
    v: NDArray[float64], gradient: NDArray[float64], step_size: float
) -> NDArray[float64]:
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert v.shape == gradient.shape
    return v + step_size * gradient


def sigmoid(t: float64) -> float64:
    """Sigmoid activation function."""
    return 1 / (1 + math.exp(-t))


def relu(t: float64) -> float64:
    """ReLU activation function."""
    return max(0, t)


def neuron(
    weights: NDArray[float64], inputs: NDArray[float64], activation: str = "sigmoid"
) -> float64:
    """Single neuron computation with selectable activation."""
    z = np.dot(weights, inputs)
    if activation == "sigmoid":
        return sigmoid(z)
    elif activation == "relu":
        return relu(z)


def feed_forward(
    neural_network: List[NDArray[float64]],
    input_vector: NDArray[float64],
    activation="sigmoid",
) -> List[NDArray[float64]]:
    """Feeds the input vector through the neural network."""
    outputs = []
    current_input = np.append(input_vector, biais)

    for layer in neural_network:
        output = np.array(
            [neuron(weights, current_input, activation) for weights in layer]
        )
        outputs.append(output)
        current_input = np.append(output, biais)  # Add bias for next layer

    return outputs


def sqerror_gradients(
    network: List[NDArray[float64]],
    input_vector: NDArray[float64],
    target_vector: NDArray[float64],
    activation: str = "sigmoid",
) -> List[NDArray[float64]]:
    """Computes the gradient of the squared error loss with respect to the neuron weights."""
    layer_outputs = feed_forward(network, input_vector, activation)
    outputs = layer_outputs[-1]
    hidden_outputs = np.append(
        layer_outputs[0], biais
    )  # Ensure consistency with bias handling
    hidden_weights = network[-1]  # Access last layer's weights
    hidden_weights_no_bias = np.array(
        [neuron_weights[:-1] for neuron_weights in hidden_weights]
    )  # Remove bias

    if activation == "sigmoid":
        output_deltas = outputs * (1 - outputs) * (outputs - target_vector)
        hidden_deltas = (
            hidden_outputs[:-1]
            * (1 - hidden_outputs[:-1])
            * np.dot(hidden_weights_no_bias.T, output_deltas)
        )
    elif activation == "relu":
        output_deltas = (outputs > 0).astype(float) * (outputs - target_vector)
        hidden_deltas = (hidden_outputs[:-1] > 0).astype(float) * np.dot(
            hidden_weights_no_bias.T, output_deltas
        )

    output_grads = np.outer(output_deltas, hidden_outputs)
    hidden_grads = np.outer(hidden_deltas, np.append(input_vector, biais))

    return [hidden_grads, output_grads]


def train_network(
    x_train, y_train, network, activation, epochs: int = 300, learning_rate: float = 0.1
):
    """Train a single-layer neural network to approximate a noisy parabolic function."""
    for epoch in range(epochs):
        total_loss = 0.0

        for x_sample, y_sample in zip(x_train, y_train):
            predicted = feed_forward(network, x_sample, activation)[-1]
            total_loss += np.sum((predicted - y_sample) ** 2)
            gradients = sqerror_gradients(
                network, x_sample, y_sample, activation=activation
            )

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

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(x_train):.4f}")

    return network


def main():
    """
    Mirar el comportamiento:
    -> muchas neuronas
    -> no suficientes puntos
    -> cambiando el activador: relu, sigmoid

    """
    n_points = 100
    val_min, val_max = -1.5, 1.5
    activation = "relu"
    epochs = 500  # 500
    learning_rate = 0.1  # .1
    x_train = np.linspace(val_min, val_max, n_points).reshape(-1, 1)
    noise = np.random.normal(0, 0.1, n_points).reshape(-1, 1)
    y_train = x_train**2 + noise
    n_neurons: int = 4
    network = [np.random.randn(n_neurons, 1 + 1), np.random.randn(1, n_neurons + 1)]
    trained_network = train_network(
        x_train,
        y_train,
        network,
        activation=activation,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    print("Final trained network:", trained_network)

    # Generate predictions for plotting
    x_grid = np.linspace(val_min, val_max, 20)
    y_pred = np.array(
        [feed_forward(trained_network, x_sample, activation)[-1] for x_sample in x_grid]
    )

    # Plot the training data and the learned model
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train, y_train, color="blue", alpha=0.5, label="Training Data")
    plt.plot(x_grid, y_pred, color="red", linewidth=2, label="Learned Model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Neural Network Approximation of Noisy Parabolic Function")
    plt.show()

    print("Final trained network:", trained_network)


if __name__ == "__main__":
    main()
