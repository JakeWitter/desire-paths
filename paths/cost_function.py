import numpy as np


def linear(x):
    return x


def sqrt(x):
    return np.sqrt(x)


def logarithmic(x):
    return np.log1p(x)


COST_Fs = {"linear": linear, "sqrt": sqrt, "logarithmic": logarithmic}
