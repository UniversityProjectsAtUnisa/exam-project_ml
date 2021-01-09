import numpy as np


def one_hot(a, n):
    e = np.eye(n)  # Identity matrix n x n
    result = e[a]
    return result


def de_one_hot(y):
    return np.argmax(y, axis=1)
