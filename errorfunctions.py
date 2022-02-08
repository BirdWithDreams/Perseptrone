import enum
import math

import numpy as np


def MSE(delta):
    return np.sum(delta ** 2, axis=1) / delta.shape[1]


def RootMSE(delta):
    return np.sqrt(MSE(delta))


def arctan(delta):
    return np.sum(np.arctan(delta) ** 2, axis=1) / delta.shape[1]


class ErrorFunctions(enum.Enum):
    MSE = MSE
    RootMSE = RootMSE
    arctan = arctan
