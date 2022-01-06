import enum
import math

import numpy as np


def MSE(delta):
    return sum(delta ** 2) / delta.size


def RootMSE(delta):
    return math.sqrt(sum(delta ** 2) / delta.size)


def arctan(delta):
    return sum(np.arctan(delta) ** 2) / delta.size


class ErrorFunctions(enum.Enum):
    MSE = MSE
    RootMSE = RootMSE
    arctan = arctan
