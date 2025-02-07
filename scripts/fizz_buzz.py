"""
Created by Analitika at 22/05/2024
contact@analitika.fr
"""
# External imports
import numpy as np
from numpy.typing import NDArray
from numpy import float64
import math


"""
Print the numbers 1 to 100, except that
if the number is divisible by 3, print "fizz";
if the number is divisible by 5, print "buzz"; and
if the number is divisible by 15, print "fizzbuzz".
"""


# For the outputs it’s not tough: there are basically four classes of outputs, so we can
# encode the output as a vector of four 0s and 1s:
def fizz_buzz_encode(x: int) -> NDArray[float64]:
    if x % 15 == 0:
        return np.array([0, 0, 0, 1])
    elif x % 5 == 0:
        return np.array([0, 0, 1, 0])
    elif x % 3 == 0:
        return np.array([0, 1, 0, 0])
    else:
        return np.array([1, 0, 0, 0])


def binary_encode(x: int, sz: int = 10) -> NDArray[np.float64]:
    """
    Encode an integer into a binary representation.

    Parameters:
    x (int): The integer to encode.
    sz (int): The size of the binary array. Default is 10.

    Returns:
    NDArray[np.float64]: A numpy array containing the binary representation.

    # Example usage
    print(binary_encode(5, 10))  # Output: [0. 0. 0. 0. 0. 0. 0. 1. 0. 1.]
    """
    if x < 0:
        raise ValueError("x must be a non-negative integer")

    binary = np.zeros(sz, dtype=np.float64)
    index = 0

    while x > 0 and index < sz:
        binary[index] = x % 2
        x = x // 2
        index += 1

    return binary


def these_functions():
    assert all(fizz_buzz_encode(2) == np.array([1, 0, 0, 0]))
    assert all(fizz_buzz_encode(6) == np.array([0, 1, 0, 0]))
    assert all(fizz_buzz_encode(10) == np.array([0, 0, 1, 0]))
    assert all(fizz_buzz_encode(30) == np.array([0, 0, 0, 1]))

    #                                        1  2  4  8 16 32 64 128 256 512
    assert all(binary_encode(0) == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert all(binary_encode(1) == np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert all(binary_encode(10) == np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0]))
    assert all(binary_encode(101) == np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 0]))
    assert all(binary_encode(999) == np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1]))


if __name__ == "__main__":
    these_functions()
    #
    # # External imports
    # import torch
    #
    # # check whether a GPU is available
    # is_gpu = torch.cuda.is_available()
    # if is_gpu:
    #     print("GPU is available")
