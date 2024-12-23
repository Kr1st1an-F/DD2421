import numpy as np
import math


def linearKernel(x, y):
    return np.dot(x, y)


def polynomialKernel(x, y, p):
    return (1 + np.dot(x, y)) ** p


def radialKernel(x, y, sigma):
    diff = np.subtract(x, y)
    return math.exp((-np.dot(diff, diff)) / (2 * sigma * sigma))
