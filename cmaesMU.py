import numpy as np
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple

_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32


class CMAES:

    def __init__(self, objective_function: str, dimensions: int, mode: str, modification_every: int):
        self._fitness = objective_function
        self._dimensions = dimensions
        self._mode = mode
        self._modification_every = modification_every
        # Initial point
        self._xmean = np.random.rand(self._dimensions)
        # Step size
        self._sigma = 1

        # Termination conditions
        self._stop_value = 1e-10
        self._stop_after = 1000 * self._dimensions ** 2

        ## Set up selection
        # Population size
        self._lambda = 10
        # Number of parents/points for recombination
        self._mu = self._lambda // 2

        # Learning rates for rank-one and rank-mu update
        self._lr_c_1 = 2 / ((self._dimensions + 1.3) ** 2 + self._mu)
        self._lr_c_mu = min(1-self._lr_c_1, 2*(self._mu-2+1/self._mu)/
                         ((self._dimensions + 2)**2 + self._mu))

        # Time constants for cumulation for step-size control
        self._c_sigma = (self._mu + 2) / (self._dimensions + self._mu + 5)
        self._damping = 1 + 2 * max(0, np.sqrt((self._mu- 1) / (self._dimensions + 1)) - 1) + self._c_sigma
        # Time constants for cumulation for rank-one update
        self._c_c = ((4 + self._mu / self._dimensions) /
                        (4 + self._dimensions + 2 * self._mu / self._dimensions))

        # E||N(0, I)||
        self._chi = np.sqrt(self._dimensions) * (1 - 1 / (4 * self._dimensions) + 1 / (21 * self._dimensions ** 2))

        # Evolution paths
        self._path_c = np.zeros(self._dimensions)
        self._path_sigma = np.zeros(self._dimensions)
        # B defines coordinate system
        self._B = None
        # D defines scaling (diagonal matrix)
        self._D = None
        # Covariance matrix
        self._C = np.eye(self._dimensions)

        # Parameter for B and D update timing
        self._generation = 0

    def generation_loop(self):
        results = []
        value = 1e32
        # for count_it in range(self._stop_after):
        for count_it in range(2):
            solutions = []
            for _ in range(self._lambda):
                # Ask a parameter
                x = self._generate_point()
                value = self._objective(x)
                solutions.append((x, value))
                print(f"#{count_it} {value} (x1={x[0]}, x2 = {x[1]})")
            if value < self._stop_value:
                break
            # Tell evaluation values
            self._tell(solutions)
            results.append([solution[1] for solution in solutions])

        plot_result(results)

    def _generate_point(self) -> np.ndarray:
        self._B, self._D = self._eigen_decomposition()
        standard_point = np.random.standard_normal(self._dimensions) # ~N(0,I)
        scaled_point = np.matmul(np.diag(self._D), standard_point) # ~N(0,D)
        rotated_point = np.matmul(self._B, scaled_point) # ~N(0,C)
        step_point = self._sigma*rotated_point # ~N(0,σ^2*C)
        moved_point = self._xmean + step_point # ~N(m, σ^2*C)
        return moved_point # ~N(m, σ^2*C)

    def _eigen_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        self._C = (self._C + self._C.T) / 2
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        self._C = np.dot(np.dot(B, np.diag(D ** 2)), B.T)
        return B, D

    def _tell(self, solutions: List[Tuple[np.ndarray, float]]) -> None:
        assert len(solutions) == self._lambda, "Must evaluate solutions with length equal to population size."
        for s in solutions:
            assert np.all(
                np.abs(s[0]) < _MEAN_MAX
            ), f"Absolute value of all solutions must be less than {_MEAN_MAX} to avoid overflow errors."

        self._generation += 1
        solutions.sort(key=lambda solution: solution[1])

        print("Mean: ", self._xmean)

        # ~ N(m, σ^2 C)
        population = np.array([s[0] for s in solutions])
        # ~ N(0, C)
        deltas = (population - self._xmean) / self._sigma

        # Selection and recombination
        selected = deltas[: self._mu]
        if self._mode == 'mean' and self._generation % self._modification_every == 0:
            selected[self._mu - 1] = np.mean(deltas, axis=0)

        deltas_sum = np.sum(selected, axis=0) / self._mu
        self._xmean += self._sigma * deltas_sum

        # Step-size control
        # C^(-1/2) = B D^(-1) B^T
        C_2 = np.matmul(np.matmul(self._B, np.diag(1 / self._D)), self._B.T)

        ps_delta = (np.sqrt(self._c_sigma * (2 - self._c_sigma) * self._mu) *
                    np.matmul(C_2, deltas_sum))
        self._path_sigma = (1 - self._c_sigma) * self._path_sigma + ps_delta

        self._sigma *= np.exp((self._c_sigma / self._damping) *
                              (np.linalg.norm(self._path_sigma) / self._chi - 1))
        self._sigma = min(self._sigma, _SIGMA_MAX)

        print("sigma: ", self._sigma)

        # Covariance matrix adaption
        self._path_c = ((1 - self._c_c) * self._path_c +
                        np.sqrt(self._c_c * (2 - self._c_c) * self._mu) * deltas_sum)

        # np.outer(v, v) == np.mat_mul(v, v.T)
        rank_one = np.outer(self._path_c, self._path_c)
        print("rank one")
        print(rank_one)
        rank_mu = np.sum([np.outer(delta, delta) for delta in deltas], axis=0)/self._mu
        print("rank mu")
        print(rank_mu)
        print("C before:")
        print(self._C)
        self._C = (
                (1 - self._lr_c_1 - self._lr_c_mu) * self._C
                + self._lr_c_1 * rank_one
                + self._lr_c_mu * rank_mu
        )
        print("C after:")
        print(self._C)

    def _objective(self, x):
        if self._fitness == 'felli':
            assert self._dimensions > 1, 'Dimension must be greater than 1.'
            return felli(x)
        elif self._fitness == 'quadratic':
            assert self._dimensions > 1, 'Dimension must be greater than 1.'
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

def felli(x: np.ndarray):
    dim = x.shape[0]
    arr = [np.power(1e6, p) for p in np.arange(0, dim) / (dim - 1)]
    return np.matmul(arr, x ** 2)

def quadratic(x: np.ndarray):
    return np.dot(x,x)
