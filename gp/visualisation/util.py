import numpy as np


def sin(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def cos(x: np.ndarray) -> np.ndarray:
    return np.cos(x)


def sin2(x: np.ndarray) -> np.ndarray:
    return np.square(np.sin(x)) - 0.5


def sincos(x: np.ndarray) -> np.ndarray:
    return np.sin(x) * np.cos(x)


def lin(x: np.ndarray) -> np.ndarray:
    return 0.1 * x + np.random.normal(0, 0.1)


def expsin(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.square(x)) * np.sin(x)


def step(x: np.ndarray) -> np.ndarray:
    return (x > 0) - 0.5


def relu(x: np.ndarray) -> np.ndarray:
    return 0.01 + np.max((0, x))
