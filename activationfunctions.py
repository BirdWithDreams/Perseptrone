import numpy as np


def sigmoid(x: np.array):
    """
    sigmoid activation function
    return the value and value of the derivative at this point
    """
    res = 1 / (1 + np.exp(-x))
    der = res * (1 - res)
    return res, der


def tanh(x: np.array):
    """
    hyperbolic tangent activation function
    return the value and value of the derivative at this point
    """
    res = np.tanh(x)
    der = 1 - res * res
    return res, der


def ReLU(x: np.array):
    """
    ReLu activation function
    return the value and value of the derivative at this point
    """
    # der = np.ones(x.shape)
    # der[x < 0] = 0

    res = np.array(x)
    res[res < 0] = 0
    return res, res > 0  # der


def LeakyReLU(x: np.array):
    """
    Leaky ReLu activation function
    return the value and value of the derivative at this point
    """
    der = np.ones(x.shape)
    der[x < 0] = 0.01

    res = np.array(x)
    res[res < 0] *= 0.01
    return res, der


def softmax(x: np.array):
    temp = np.exp(x)
    return temp / temp.sum(), 1 / len(temp)
