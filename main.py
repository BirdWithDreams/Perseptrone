import numpy as np
from perceptron import Perceptron
from errorfunctions import ErrorFunctions

with open('mnist_train.csv', 'r') as data_file:
    data = data_file.readlines()[:1000]
input = []
answers = []

for _data in data:
    input.append(np.array(list(map(int, _data[2:].split(',')))) / 255)
    answ = np.zeros(10)
    answ[int(_data[0])] = 1
    answers.append(answ)

perc = Perceptron('Test', 0.1, 100, error_func=ErrorFunctions.RootMSE)
perc.addLayer(200, perc.actFunctions.tanh, range_=(-0.01, 0.01))
perc.addLayer(10, perc.actFunctions.softmax, range_=(-0.01, 0.01))
perc.set_data(input, answers)
perc.init()
perc.start()
perc.test_loop()
