import numpy as np


def sin(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


def sin2(x):
    return np.square(np.sin(x)) - 0.5


def sincos(x):
    return np.sin(x) * np.cos(x)


def lin(x):
    return 0.1 * x + np.random.normal(0, 0.1)


def expsin(x):
    return np.exp(-np.square(x)) * np.sin(x)


def step(x):
    return (x > 0) - 0.5


def relu(x):
    return 0.01 + np.max((0, x))
