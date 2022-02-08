import os

import numpy as np


class FullConnectedLayer:
    def __init__(self, perceptron, size: tuple, activation_function):
        self._size = size
        self._neurons = np.ones((perceptron._batch_size, self._size + 1))
        self._derivative = np.ones(self._size)
        self._function = activation_function

        self._perceptron = perceptron

    def __repr__(self):
        return f"Full Connected Layer: {self._size=}, {self._function=} with weights:\n\t" + str(self.weights)

    def __str__(self):
        return f"Full Connected Layer: size - {self._size}, " \
               f"function - {self._function.__name__}"

    def set_weights(self, size):
        self._weights = np.random.uniform(-1, 1, size)  # TODO: add range of random weights generation

        if self._next is not None:
            self._next.set_weights((self.size + 1, self._next.size))

    def set_next(self, layer):
        self._next = layer

    def activation(self, _input: np.array):
        self._input = np.array(_input)
        _n = self._input @ self._weights

        self._neurons[:, :-1], self._derivative = self._function(_n)
        # self._neurons = np.concatenate((self._neurons, np.ones((1, self._neurons.shape[0]))), axis=1)
        if self._next is not None:
            return self._next.activation(self._neurons)
        else:
            return self._neurons

    def back_propagation(self, delta):
        if self._next is not None:
            _delta = self._derivative * self._next.back_propagation(delta)
        else:
            _delta = self._derivative * delta

        _del = _delta @ self._weights.T

        self._weights -= self._perceptron.a * self._input.T @ _delta
        return _del[:, :-1]

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
        self._weights = np.copy(value)
