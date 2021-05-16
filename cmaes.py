import numpy as np
import math
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple

_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32


class CMAES:

    def __init__(self, objective_function: str, dimensions: int):
        self._fitness = objective_function
        self._dimension = dimensions
        # Initial point
        self._xmean = np.random.rand(self._dimension)
        # Step size
        self._sigma = 0.05
        self._stop_value = 1e-10
        self._stop_after = 1000 * self._dimension ** 2

        # Set up selection
        # Population size
        self._lambda = int(4 + np.floor(3 * np.log(self._dimension)))
        # Number of parents/points for recombination
        self._mu = self._lambda // 2
        # Recombination weights
        weights_prime = np.log(self._mu + 0.5) - np.log(np.arange(1, self._lambda + 1))
        # Variance effective number of parents
        self._mu_effective = sum(weights_prime[:self._mu]) ** 2 / sum(weights_prime[:self._mu] ** 2)
        self._mu_effective_minus = sum(weights_prime[self._mu:]) ** 2 / sum(weights_prime[self._mu:] ** 2)

        # Learning rates for rank-one and rank-mu update
        self._lr_c1 = 2 / ((self._dimension + 1.3) ** 2 + self._mu_effective)
        self._lr_c_mu = (2 * (self._mu_effective - 2 + 1 / self._mu_effective) /
                         ((self._dimension + 2) ** 2 + 2 * self._mu_effective / 2))

        min_alpha = min(
            1 + self._lr_c1 / self._lr_c_mu,
            1 + (2 * self._mu_effective_minus / (self._mu_effective + 2)),
            (1 - self._lr_c1 - self._lr_c_mu) / (self._dimension * self._lr_c_mu)
        )
        positive_sum = np.sum(weights_prime[weights_prime > 0])
        negative_sum = np.sum(np.abs(weights_prime[weights_prime < 0]))

        self._weights = np.where(
            weights_prime >= 0,
            1 / positive_sum * weights_prime,
            min_alpha / negative_sum * weights_prime,
        )

        # Time constants for cumulation for step-size control
        self._time_sigma = (self._mu_effective + 2) / (self._dimension + self._mu_effective + 5)
        self._damping = 1 + 2 * max(0, np.sqrt((self._mu_effective - 1) / (self._dimension + 1)) - 1) + self._time_sigma
        # Time constants for cumulation for rank-one update
        self._time_c = ((4 + self._mu_effective / self._dimension) /
                        (4 + self._dimension + 2 * self._mu_effective / self._dimension))

        # E||N(0, I)||
        self._chi = np.sqrt(self._dimension) * (1 - 1 / (4 * self._dimension) + 1 / (21 * self._dimension ** 2))

        # Evolution paths
        self._path_c = np.zeros(self._dimension)
        self._path_sigma = np.zeros(self._dimension)
        # B defines coordinate system
        self._B = None
        # D defines scaling (diagonal matrix)
        self._D = None
        # Covariance matrix
        self._C = np.eye(self._dimension)

        self._bounds = None
        self._n_max_resampling = 100

        # Parameter for B and D update timing
        self._generation = 0

        # Termination criteria
        self._tolx = 1e-12 * self._sigma
        self._tolxup = 1e4
        self._tolfun = 1e-12
        self._tolconditioncov = 1e14

    def generation_loop(self):
        results = []
        value = 1e32
        for count_it in range(self._stop_after):
            solutions = []
            for _ in range(self._lambda):
                # Ask a parameter
                x = self._sample_solution()

                value = self.objective(x)
                solutions.append((x, value))
                print(f"#{count_it} {value} (x1={x[0]}, x2 = {x[1]})")
            if value < self._stop_value:
                break
            # Tell evaluation values.
            self.tell(solutions)
            results.append([x[1] for x in solutions])

        plot_result(results)

    def _sample_solution(self) -> np.ndarray:
        self._B, self._D = self._eigen_decomposition()
        arz = np.random.standard_normal(self._dimension)
        return self._xmean + self._sigma * np.matmul(np.matmul(self._B, np.diag(self._D)), arz)

    def _eigen_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        self._C = (self._C + self._C.T) / 2
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        self._C = np.dot(np.dot(B, np.diag(D ** 2)), B.T)

        return B, D

    def tell(self, solutions: List[Tuple[np.ndarray, float]]) -> None:
        assert len(solutions) == self._lambda, "Must evaluate solutions with length equal to population size."
        for s in solutions:
            assert np.all(
                np.abs(s[0]) < _MEAN_MAX
            ), f"Absolute value of all solutions must be less than {_MEAN_MAX} to avoid overflow errors."

        self._generation += 1
        solutions.sort(key=lambda solution: solution[1])

        # ~ N(m, σ^2 C)
        population = np.array([s[0] for s in solutions])
        # ~ N(0, C)
        y_k = (population - self._xmean) / self._sigma

        # Selection and recombination
        y_w = np.sum(y_k[: self._mu].T * self._weights[: self._mu], axis=1)
        self._xmean += self._sigma * y_w

        # Step-size control
        # C^(-1/2) = B D^(-1) B^T
        C_2 = np.matmul(np.matmul(self._B, np.diag(1 / self._D)), self._B.T)

        ps_delta = (np.sqrt(self._time_sigma * (2 - self._time_sigma) * self._mu_effective) *
                    np.matmul(C_2, y_w))
        self._path_sigma = (1 - self._time_sigma) * self._path_sigma + ps_delta

        norm_p_sigma = np.linalg.norm(self._path_sigma)

        self._sigma *= np.exp((self._time_sigma / self._damping) *
                              (np.linalg.norm(self._path_sigma) / self._chi - 1))
        self._sigma = min(self._sigma, _SIGMA_MAX)

        # Covariance matrix adaption
        h_sigma = self._heaviside_function(norm_p_sigma)

        self._path_c = ((1 - self._time_c) * self._path_c +
                        h_sigma * np.sqrt(self._time_c * (2 - self._time_c) * self._mu_effective) * y_w)

        w_io = self._weights * np.where(
            self._weights >= 0,
            1,
            self._dimension / (np.linalg.norm(np.matmul(C_2, y_k.T), axis=0) ** 2 + _EPS),
        )

        delta_h_sigma = (1 - h_sigma) * self._time_c * (2 - self._time_c)  # (p.28)

        # np.outer(v, v) == np.mat_mul(v, v.T)
        rank_one = np.outer(self._path_c, self._path_c)
        rank_mu = np.sum(np.array([w * np.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0)
        self._C = (
                (1 + self._lr_c1 * delta_h_sigma - self._lr_c1 - self._lr_c_mu * np.sum(self._weights)) * self._C
                + self._lr_c1 * rank_one
                + self._lr_c_mu * rank_mu
        )

    def _heaviside_function(self, norm_p_sigma):
        return int(norm_p_sigma / np.sqrt(1 - (1 - self._time_sigma) ** (2 * (self._generation + 1))) / self._chi
                   < 1.4 + 2 / (self._dimension + 1))

    def objective(self, x):
        if self._fitness == 'felli':
            assert self._dimension > 1, 'Dimension must be greater than 1.'
            return felli(x)
        elif self._fitness == 'quadratic':
            assert self._dimension == 2, 'Invalid dimension for quadratic function.'
            return quadratic(x)
        raise Exception('Invalid objective function chosen')


def plot_result(results):
    results = [np.max(list) for list in results]
    x_axis = np.arange(len(results))
    plt.plot(x_axis, results)
    plt.xlabel('Timestep')
    plt.ylabel('Obj fun')
    plt.grid()
    plt.show()


def generate_random_matrix(x, y):
    return [np.random.rand(y) for _ in range(x)]


def felli(x: np.ndarray):
    dim = x.shape[0]
    arr = [np.power(1e6, p) for p in np.arange(0, dim) / (dim - 1)]
    return np.matmul(arr, x ** 2)


def quadratic(x: np.ndarray):
    return (x[0] - 3) ** 2 + (10 * (x[1] + 2)) ** 2
