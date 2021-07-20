import numpy as np
import time
import matplotlib.pyplot as plt

from typing import List, Tuple

_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32


class CMAES:

    def __init__(self, objective_function: str, dimensions: int, mode: str, lambda_arg: int = 100):
        self._fitness = objective_function
        self._dimension = dimensions
        self._mode = mode
        # Initial point
        self._xmean = 5 * np.random.rand(self._dimension)
        # Step size
        self._sigma = 10
        self._stop_value = 1e-10
        self._stop_after = 100 * self._dimension ** 2

        # Set up selection
        # Population size
        assert lambda_arg > 2, "Lambda must be greater than 2"
        self._lambda = lambda_arg
        # Number of parents/points for recombination
        self._mu = self._lambda // 2

        # Learning rates for rank-one and rank-mu update
        self._lr_c1 = 2 / ((self._dimension + 1.3) ** 2 + self._mu)
        self._lr_c_mu = (2 * (self._mu- 2 + 1 / self._mu) /
                         ((self._dimension + 2) ** 2 + 2 * self._mu/ 2))

        # Time constants for cumulation for step-size control
        self._time_sigma = (self._mu+ 2) / (self._dimension + self._mu+ 5)
        self._damping = 1 + 2 * max(0, np.sqrt((self._mu- 1) / (self._dimension + 1)) - 1) + self._time_sigma
        # Time constants for cumulation for rank-one update
        self._time_c = ((4 + self._mu/ self._dimension) /
                        (4 + self._dimension + 2 * self._mu/ self._dimension))

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

        # Parameter for B and D update timing
        self._generation = 0

        self._results = []
        self._best_value = float('inf')

    def generation_loop(self):
        self._results = []
        for count_it in range(self._stop_after):
            self._B, self._D = self._eigen_decomposition()
            solutions = []
            value_break_condition = False

            for _ in range(self._lambda):
                x = self._sample_solution()

                value = self.objective(x)
                self._best_value = min(value, self._best_value)
                if value < self._stop_value:
                    value_break_condition = True
                    self._results.append((count_it, value))
                    break
                solutions.append((x, value))

            if value_break_condition:
                break

            # Tell evaluation values.
            assert len(solutions) == self._lambda, "There must be exatcly lambda points generated"
            self.tell(solutions)
            self._results.append((count_it, self._best_value))

    def _sample_solution(self) -> np.ndarray:
        std = np.random.standard_normal(self._dimension)
        return self._xmean + self._sigma * np.matmul(np.matmul(self._B, np.diag(self._D)), std)

    def _eigen_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
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
        selected = y_k[: self._mu]
        y_w = np.mean(selected, axis=0)
        self._xmean += self._sigma * y_w

        #debug:
        x1 = [point[0] for point in y_k]
        x2 = [point[1] for point in y_k]
        plt.scatter(x1, x2)
        x1 = [point[0] for point in selected]
        x2 = [point[1] for point in selected]
        plt.scatter(x1, x2)
        plt.grid()
        plt.show(block = False)
        plt.pause(0.5)
        plt.clf()
        plt.cla()
        #debug

        # Step-size control
        # C^(-1/2) = B D^(-1) B^T
        C_2 = np.matmul(np.matmul(self._B, np.diag(1 / (self._D + _EPS))), self._B.T)

        ps_delta = (np.sqrt(self._time_sigma * (2 - self._time_sigma) * self._mu) *
                    np.matmul(C_2, y_w))
        self._path_sigma = (1 - self._time_sigma) * self._path_sigma + ps_delta

        self._sigma *= np.exp((self._time_sigma / self._damping) *
                              (np.linalg.norm(self._path_sigma) / self._chi - 1))
        self._sigma = min(self._sigma, _SIGMA_MAX)

        # Covariance matrix adaption
        self._path_c = ((1 - self._time_c) * self._path_c +
                        np.sqrt(self._time_c * (2 - self._time_c) * self._mu) * y_w)

        # np.outer(v, v) == np.mat_mul(v, v.T)
        rank_one = np.outer(self._path_c, self._path_c)
        rank_mu = np.mean([np.outer(d, d) for d in selected], axis=0)
        self._C = (
                (1 - self._lr_c1 - self._lr_c_mu) * self._C
                + self._lr_c1 * rank_one
                + self._lr_c_mu * rank_mu
        )
        print("rank one", rank_one)
        print("rank mu", rank_mu)

    def objective(self, x):
        assert self._dimension > 0, 'Number of dimensions must be greater than 0.'
        if self._fitness == 'felli':
            return felli(x)
        elif self._fitness == 'quadratic':
            return quadratic(x)
        elif self._fitness == 'bent':
            return bent_cigar(x)
        elif self._fitness == 'rastrigin':
            return rastrigin(x)
        elif self._fitness == 'rosenbrock':
            return rosenbrock(x)
        raise Exception('Invalid objective function chosen')

    def ecdf(self, targets: np.array):
        assert self._results != [], "Can't plot results, must run the algorithm first"
        ecdf = []
        for result in self._results:
            passed = 0
            for target in targets:
                if result[1] <= target:
                    passed += 1
            ecdf.append(passed / len(targets))
        return ecdf

    def plot_result(self):
        assert self._results != [], "Can't plot results, must run the algorithm first"
        x_axis = [it for (it, _) in self._results]
        y_axis = [value for (_, value) in self._results]
        plt.xlabel('Iterations')
        plt.ylabel('Best objective value')
        plt.yscale('log')
        plt.grid()
        plt.scatter(x_axis, y_axis)

def felli(x: np.ndarray) -> float:
    dim = x.shape[0]
    if dim == 1:
        return quadratic(x)
    arr = [np.power(1e6, p) for p in np.arange(0, dim) / (dim - 1)]
    return float(np.matmul(arr, x ** 2))


def quadratic(x: np.ndarray) -> float:
    return float(np.dot(x, x))


def bent_cigar(x: np.ndarray) -> float:
    return x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2)


def rastrigin(x: np.ndarray) -> float:
    return float(np.sum(x ** 2 + -10 * (np.cos(2 * np.pi * x)) + 10))


def rosenbrock(x: np.ndarray) -> float:
    return sum([100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2 for i in range(x.shape[0] - 1)])
