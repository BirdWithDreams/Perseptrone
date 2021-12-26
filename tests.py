import numpy as np
import math


def func(x):
    return 1 / (1 + math.exp(-x))


# input = np.array([0, 1, 2])
# weights = np.random.uniform(-1, 1, (3, 4))
#
# _out = input.reshape((3, 1)) * weights
# out = func(_out.sum(axis=1))

vec = np.array([1, 0, -1])
mat = np.array([[1, -1, 0], [0, 1, 1], [1, 2, 0]])
h = vec*mat
vec[vec > 0] = 2
print(vec)
