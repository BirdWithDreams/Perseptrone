import json
import math

import numpy as np

import activationfunctions as af
from fullconnectedlayer import FullConnectedLayer


class Perceptron:
    actFunctions = af

    # this is main class
    # The class for perseptrone

    def __init__(self, learning_speed: float = 1, quantity: int = 1, error_func=None, name: str = None):
        self.layers = []
        self.average_error = [0, 0]

        self._a = learning_speed
        self._quantity = quantity
        self._name = name
        self._error_func = error_func

    def addLayer(self, size: tuple, activation_function):
        layer = FullConnectedLayer(self, size, activation_function)
        if self.layers:
            self.layers[-1].set_next(layer)

        self.layers.append(layer)

    def set_data(self, data, answers):
        self.data = np.array(data)
        self.answers = np.array(answers)

    def init(self):
        self.layers[-1].set_next(None)
        self.init_weights()

    def init_weights(self):
        self.layers[0].set_weights((self.layers[0].size, self.layers[0].size))

    def start(self):
        if not self.layers:
            raise ValueError('You have to add a layer!')

        if self._error_func is None:
            raise AttributeError('You must specify the method of calculation error!')

        self.main_loop()

    def main_loop(self):
        for i in range(self.quantity):
            for iter, _data in enumerate(self.data):
                self.learning(_data, self.answers[iter])

            self.average_error = [0, 0]

    def learning(self, _input, output):
        res = self.layers[0].activation(_input)
        delta = res[:-1] - output
        self.layers[0].back_propagation(delta)

        try:
            error = self._error_func(delta)
            self.calculate_average_error(error)
        except Exception as e:
            print(e)

    def calculate_average_error(self, error):
        self.average_error[1] += 1
        self.average_error[0] = (self.average_error[0] + error) / self.average_error[1]

    def test_loop(self):
        for iter, _data in enumerate(self.data):
            output = self.layers[0].activation(_data)[:-1]
            res = f"Input: {_data}\nOutput: {output}\n"
            print(res)

            delta = output[:-1] - self.answers[iter]
            error = self._error_func(delta)
            self.calculate_average_error(error)

        print(f"Error: {self.average_error[0]}")

    def save(self):
        if self.name is not None:
            names = [f'weights_{i}_{i + 1}' for i in range(len(self.layers))]
            perc = {
                'name': self.name,
                'weights file name': self.name + ' weights.npz',
                'layers': [{"size": layer.size,
                            "file name": name,
                            "activation function": layer._function.__name__} for layer, name in
                           zip(self.layers, names)],
            }

            with open(self.name + '_perceptron.json', "w") as file:
                json.dump(perc, file, indent=4)

            np.savez(perc['weights file name'],
                     **dict((name, layer.weights) for name, layer in zip(names, self.layers)))

        else:
            raise AttributeError("If you want to save it, you must give it the name!")

    @classmethod
    def load(cls, file_name):
        with open(file_name, "r") as file:
            perc = json.load(file)

        perceptron = cls(name=perc["name"])
        file = np.load(perc['weights file name'])
        for layer in perc["layers"]:
            perceptron.addLayer(layer["size"], cls.actFunctions.__dict__[layer["activation function"]])
            perceptron.layers[-1].weights = file[layer["file name"]]

            if len(perceptron.layers) > 1:
                perceptron.layers[-2].set_next(perceptron.layers[-1])

            perceptron.layers[-1].set_next(None)

        return perceptron

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        self._a = a
        for item in self.__dict__.items(): print(item)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def quantity(self):
        return self._quantity

    @quantity.setter
    def quantity(self, value):
        self._quantity = value


class ErrorFunctions:
    @staticmethod
    def MSE(delta):
        return sum(delta ** 2) / delta.size

    @staticmethod
    def RootMSE(delta):
        return math.sqrt(sum(delta ** 2) / delta.size)

    @staticmethod
    def Arctan(delta):
        return sum(np.arctan(delta) ** 2) / delta.size
