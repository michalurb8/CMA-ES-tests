import numpy as np
from typing import Callable

def elliptic(x: np.ndarray) -> float:
    dim = x.shape[0]
    if dim == 1:
        return float(np.dot(x, x))
    arr = [np.power(1e6, p) for p in np.arange(0, dim) / (dim - 1)]
    return float(np.matmul(arr, x ** 2))

def quadratic(x: np.ndarray) -> float:
    return float(np.dot(x, x))

def bent_cigar(x: np.ndarray) -> float:
    return x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2)

def rastrigin(x: np.ndarray) -> float:
    return float(np.sum(x ** 2 + -10 * (np.cos(2 * np.pi * x)) + 10))

def rosenbrock(x: np.ndarray) -> float:
    return sum([100 * ((x[i]+1) ** 2 - (x[i + 1]+1)) ** 2 + x[i] ** 2 for i in range(x.shape[0] - 1)])

def ackley(x: np.ndarray) -> float:
    exp1 = -20 * np.exp(-0.2*np.sqrt(np.sum(x**2)/len(x))) + 20
    exp2 = -1  * np.exp(np.sum(np.cos(2*np.pi*x)/len(x))) + np.e
    return exp1 + exp2



def Get_by_name(name: str) -> Callable:
    if name == "elliptic":
        return elliptic
    elif name == "quadratic":
        return quadratic
    elif name == "bent_cigar":
        return bent_cigar
    elif name == "rastrigin":
        return rastrigin
    elif name == "rosenbrock":
        return rosenbrock
    elif name == "ackley":
        return ackley