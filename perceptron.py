import json

import numpy as np

import activationfunctions as af
from errorfunctions import ErrorFunctions
from fullconnectedlayer import FullConnectedLayer


class Perceptron:
    actFunctions = af

    def __init__(self,
                 name: str = None,
                 learning_speed: float = 1,
                 quantity: int = 1,
                 error_func=None,
                 batch_size=1):

        self.layers = []
        self.average_error = [0, 0]

        self._a = learning_speed
        self._quantity = quantity
        self._name = name
        self._error_func = error_func

        self._batch_size = batch_size
        self._batch_start = None
        self._batch_end = None

    def __repr__(self):
        return f"Perceptron '{self.name}' with layers: \n\t" + "\n\t".join(repr(layer) for layer in self.layers)

    def __str__(self):
        return f"Perceptron '{self.name}' with layers: \n\t" + "\n\t".join(str(layer) for layer in self.layers)

    def __call__(self, data, *, start=0):
        return self.layers[start].activation(data)

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
        self.layers[0].set_weights((len(self.data[0]), self.layers[0].size))

    def start(self):
        if not self.layers:
            raise ValueError('You have to add a layer!')

        if self._error_func is None:
            raise AttributeError('You must specify the method of calculation error!')

        self.main_loop()

    def main_loop(self):
        for i in range(self.quantity):
            for iter in range(int(len(self.data) / self._batch_size)):
                self._batch_start = iter * self._batch_size
                self._batch_end = (iter + 1) * self._batch_size
                data = self.data[self._batch_start:self._batch_end]

                self.learning(data, self.answers[self._batch_start:self._batch_end])

                self.average_error = [0, 0]

    def learning(self, _input, output):
        res = self.layers[0].activation(_input)
        delta = (res[:, :-1] - output) / self._batch_size
        self.layers[0].back_propagation(delta)

        try:
            error = self._error_func(delta)
            self.calculate_average_error(error)
            print(error)
        except Exception as e:
            print(e)

    def calculate_average_error(self, error):
        self.average_error[1] += 1
        self.average_error[0] = (self.average_error[0] + error) / self.average_error[1]

    def test_loop(self):
        correct_cnt = 0
        for iter, _ in enumerate(self.data):
            output = self.layers[0].activation(self.data[iter:iter+1])[:, :-1]
            res = f"Input: {self.data[iter:iter+1]}\nOutput: {output}\n"
            # print(res)

            delta = output - self.answers[iter:iter+1]
            error = self._error_func(delta)
            # print(error)
            correct_cnt += int(output.argmax() == self.answers[iter].argmax())
            self.calculate_average_error(error)

        print(f"Error: {self.average_error[0]}\n"
              f"Test-Acc: {correct_cnt / len(self.data)}")

    def save(self):
        if self.name is not None:
            names = [f'weights_{i}_{i + 1}' for i in range(len(self.layers))]
            perc = {
                'name': self.name,
                'error_function': self._error_func.__name__,
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
        perceptron._error_func = ErrorFunctions.__dict__[perc['error_function']]
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

def copy(other):
    _args = (other.a, other.quantity, other._error_func, other.name)
    obj = other.__class__(*_args)

    for layer in other.layers:
        obj.addLayer(layer.size, layer._function)
        obj.layers[-1]._weights = layer.weights

    obj.set_data(other.data, other.answers)
    obj.layers[-1].set_next(None)

    return obj
