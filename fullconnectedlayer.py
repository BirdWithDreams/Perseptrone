import os

import numpy as np


class FullConnectedLayer:
    def __init__(self, perceptron, size: tuple, activation_function):
        self._size = size
        self._neurons = np.ones(self._size + 1)
        self._derivative = np.ones(self._size)
        self._function = activation_function

        self._perceptron = perceptron

    def set_weights(self, size):
        self._weights = np.random.uniform(-1, 1, size)

        if self._next is not None:
            self._next.set_weights((self._next.size, self.size + 1))

    def set_next(self, layer):
        self._next = layer

    def activation(self, _input: np.array):
        self._input = np.array(_input)
        _n = self._weights @ self._input

        self._neurons[:-1], self._derivative = self._function(_n)
        if self._next is not None:
            return self._next.activation(self._neurons)
        else:
            return self._neurons

    def back_propagation(self, delta):
        if self._next is not None:
            _delta = self._next.back_propagation(delta)
            _delta = _delta.reshape((-1, 1))
        else:
            _delta = delta.reshape((-1, 1))

        _del = self._weights * _delta
        _del = _del.sum(axis=0)

        self._weights -= self._perceptron.a * self._input * _delta * self._derivative
        return _del[:-1]

    def save_to_file(self, name, zip_file):
        np.savez(name, self._weights)
        zip_file.write(name + '.npy')
        os.remove(name + '.npy')

    @property
    def size(self):
        return self._size

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value
