"""
Created by Eduardo Cepeda at 25/04/2024
eduardo@cepeda.fr
"""
# External imports
from typing import Callable, Iterator
import numpy as np
from numpy.typing import NDArray
from numpy import float64
import random

# Internal imports


def sum_of_squares(v: NDArray[float64]) -> float:
    """Computes $||v||_2^2$, the sum of squared elements in v"""
    return v @ v


def difference_quotient(f: Callable[[float], float], x: float, h: float) -> float:
    """Returns an estimate of the first derivative of f: df/dx at x"""
    return (f(x + h) - f(x)) / h


def square(x: float) -> float:
    """Returns x squared"""
    # this will be used as example input in an operator
    return x * x


def derivative(x: float) -> float:
    """Returns the derivative of the $f(x) = x^2$ function"""
    return 2 * x


def partial_difference_quotient(
    f: Callable[[NDArray[float64]], float], v: NDArray[float64], i: int, h: float
) -> float:
    """Returns the i-th partial difference quotient of f at v"""
    # Create a copy of v, the original array v is not altered
    # maintaining the functional approach of not modifying inputs
    w = np.copy(v)
    # Add h to just the ith element of v
    w[i] += h

    return (f(w) - f(v)) / h


def estimate_gradient(
    f: Callable[[NDArray[float64]], float], v: NDArray[float64], h: float = 0.0001
) -> NDArray[float64]:
    return np.array([partial_difference_quotient(f, v, i, h) for i in range(len(v))])


def gradient_step(
    v: NDArray[float64], gradient: NDArray[float64], step_size: float
) -> NDArray[float64]:
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert v.shape[0] == gradient.shape[0]
    step = step_size * gradient
    return v + step


def sum_of_squares_gradient(v: NDArray[float64]) -> NDArray[float64]:
    return 2 * v


def linear_gradient(x: float, y: float, theta: NDArray[float64]) -> NDArray[float64]:
    slope, intercept = theta
    predicted = slope * x + intercept  # The prediction of the model.
    error = predicted - y  # error is (predicted - actual)
    # squared_error = error**2  # We'll minimize squared error
    gradient = np.array([2 * error * x, 2 * error])  # using its gradient.
    return gradient


def minibatches(
    dataset: NDArray[float64], batch_size: int, shuffle: bool = True
) -> Iterator[NDArray[float64]]:
    """Generates `batch_size`-sized minibatches from the dataset"""
    # Start indexes 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle:
        random.shuffle(batch_starts)  # shuffle the batches

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]


def compare_estimates():
    # how good is our derivative estimate?
    xs = range(-10, 11)
    actuals = [derivative(x) for x in xs]
    estimates = [difference_quotient(square, x, h=1e-3) for x in xs]

    # plot to show they're basically the same
    import matplotlib.pyplot as plt

    plt.title("Actual Derivatives vs. Estimates")
    plt.plot(xs, actuals, "rx", label="Actual")  # red  x
    plt.plot(xs, estimates, "b+", label="Estimate")  # blue +
    plt.legend(loc=9)
    plt.show()
    # plt.close()


def main():

    compare_estimates()

    # "Using the Gradient" example
    # ============================

    # We'll minimize $x^2$ for $x \in R^3$
    # pick a random starting point
    v = np.random.uniform(-10, 10, 3)

    for epoch in range(1000):
        grad = sum_of_squares_gradient(v)  # compute the gradient at v
        v = gradient_step(v, grad, -1e-2)  # take a negative gradient step
        print(epoch, v)

    assert np.linalg.norm(v) < 1e-6, "v should be close to 0"

    # First "Using Gradient Descent to Fit Models" example
    # 1. Start with a random value for theta.
    # 2. Compute the mean of the gradients.
    # 3. Adjust theta in that direction.
    # 4. Repeat.
    # ====================================================

    # x ranges from -50 to 49, y is always 20 * x + 5
    params = [20, 5]  # slope, intercept
    inputs = np.array([(x, params[0] * x + params[1]) for x in range(-50, 50)])
    # Start with random values for slope and intercept.
    theta = np.random.uniform(-1, 1, 2)

    learning_rate = 1e-3
    # we can also add a tolerance parameter to stop early if we're converging
    tol = 1e-2
    for epoch in range(5000):
        # Compute the mean of the gradients
        grads_list = [linear_gradient(x, y, theta) for x, y in inputs]
        grad = np.vstack(grads_list).mean(axis=0)
        # Take a step in that direction
        theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
        # Check if we've converged
        if np.linalg.norm(theta - params) < tol:
            print(f"Converged in {epoch} epochs")
            break

    slope, intercept = theta
    assert np.linalg.norm(theta - params) < tol, f"params should be {params}"
    # Check if we've converged element-wise (there's no reason, we stopped when L2-convergence)
    assert abs(slope - params[0]) < tol, f"slope should be about {params[0]}"
    assert abs(intercept - params[1]) < tol, f"intercept should be about {params[1]}"

    # Minibatch gradient descent example
    # ==================================
    theta = np.random.uniform(-1, 1, 2)

    for epoch in range(1000):
        for batch in minibatches(inputs, batch_size=20):
            grads_list = [linear_gradient(x, y, theta) for x, y in batch]
            grad = np.vstack(grads_list).mean(axis=0)
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
        # Check if we've converged
        if np.linalg.norm(theta - params) < tol:
            print(f"Converged in {epoch} mini-batches epochs")
            break

    slope, intercept = theta
    assert abs(slope - params[0]) < tol, f"slope should be about {params[0]}"
    assert abs(intercept - params[1]) < tol, f"intercept should be about {params[1]}"
    assert np.linalg.norm(theta - np.array([20, 5])) < tol, f"params should be {params}"

    # Stochastic gradient descent example
    # On this problem, stochastic gradient descent finds the optimal parameters in a much
    # smaller number of epochs. But there are always tradeoffs. Basing gradient steps on
    # small minibatches (or on single data points) allows you to take more of them, but the
    # gradient for a single point might lie in a very different direction from the gradient for
    # the dataset as a whole.
    # ===================================
    theta = np.random.uniform(-1, 1, 2)
    print("initial:", theta)
    for epoch in range(100):
        for x, y in inputs:
            grad = linear_gradient(x, y, theta)
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
        # Check if we've converged
        if np.linalg.norm(theta - params) < tol:
            print(f"Converged in {epoch} 1 mini-batches (stochastic) epochs")
            break

    slope, intercept = theta
    assert abs(slope - params[0]) < tol, f"slope should be about {params[0]}"
    # remark how the tolerance was relaxed for element-wise checks
    assert (
        abs(intercept - params[1]) < tol * 10
    ), f"intercept should be about {params[1]}"
    assert (
        np.linalg.norm(theta - np.array([20, 5])) < tol * 10
    ), f"params should be {params}"


if __name__ == "__main__":
    main()
