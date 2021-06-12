import numpy as np
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple

_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32
_TARGETS = np.array([10**i for i in range(-10, 10)])


class CMAES:

    def __init__(self, objective_function: str, dimensions: int, mode: str, modification_every: int):
        self._fitness = objective_function
        self._dimension = dimensions
        self._mode = mode
        self._modification_every = modification_every
        self._set_initial_params()
    
    def _set_initial_params(self):
        # Initial point
        self._xmean = np.random.rand(self._dimension)
        # Step size
        self._sigma = 10
        self._stop_value = 1e-10
        self._stop_after = 1000 * self._dimension ** 2

        # Set up selection
        # Population size
        self._lambda = int(4 + np.floor(3 * np.log(self._dimension)))
        # Number of parents/points for recombination
        self._mu = self._lambda // 2

        # Learning rates for rank-one and rank-mu update
        self._lr_c1 = 2 / ((self._dimension + 1.3) ** 2 + self._mu)
        self._lr_c_mu = (2 * (self._mu- 2 + 1 / self._mu) /
                         ((self._dimension + 2) ** 2 + 2 * self._mu/ 2))

        # Time constants for cumulation for step-size control
        self._time_sigma = (self._mu + 2) / (self._dimension + self._mu + 5)
        self._damping = 1 + 2 * max(0, np.sqrt((self._mu - 1) / (self._dimension + 1)) - 1) + self._time_sigma
        # Time constants for cumulation for rank-one update
        self._time_c = ((4 + self._mu / self._dimension) /
                        (4 + self._dimension + 2 * self._mu / self._dimension))

        # E||N(0, I)||
        self._chi = np.sqrt(self._dimension) * (1 - 1 / (4 * self._dimension) + 1 / (21 * self._dimension ** 2))

        # Evolution paths
        self._path_c = np.zeros(self._dimension)
        self._path_sigma = np.zeros(self._dimension)
        # B defines coordinate system
        self._B = np.eye(self._dimension)
        # D defines scaling (diagonal matrix)
        self._D = np.eye(self._dimension)
        # Covariance matrix
        self._C = np.eye(self._dimension)

        # Parameter for B and D update timing
        self._generation = 0

        #store best found value so far
        self._best_value = float('inf')
        self.results = []


    def generation_loop(self):
        value = float('inf')
        self.results = []
        for count_it in range(self._stop_after):
            self._B, self._D = self._eigen_decomposition()
            solutions = []
            value_break_condition = False
            for _ in range(self._lambda):
                # Ask a parameter
                x = self._sample_solution()

                value = self._objective(x)
                self._best_value = min(value, self._best_value)

                solutions.append((x, value))
                if value < self._stop_value:
                    value_break_condition = True
                    self.results.append((count_it, self._best_value))
                    break
            if value_break_condition == True:
                break
            # Tell evaluation values.
            self.tell(solutions)
            self.results.append((count_it, self._best_value))
        else:
            print("Iteration limit reached.")

    def _sample_solution(self) -> np.ndarray:
        standard_point = np.random.standard_normal(self._dimension) # ~N(0,I)
        scaled_point = np.matmul(np.diag(self._D), standard_point) # ~N(0,D)
        rotated_point = np.matmul(self._B, scaled_point) # ~N(0,C)
        step_point = self._sigma*rotated_point # ~N(0,σ^2*C)
        moved_point = self._xmean + step_point # ~N(m, σ^2*C)
        return moved_point # ~N(m, σ^2*C)

    def _eigen_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        # self._C = (self._C + self._C.T) / 2 # WHY
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        # self._C = np.dot(np.dot(B, np.diag(D ** 2)), B.T) # WHY
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
        if self._mode == 'mean' and self._generation % self._modification_every == 0:
            selected[self._mu - 1] = np.mean(y_k, axis=0)
        y_w = np.sum(selected, axis=0)/self._mu
        self._xmean += self._sigma * y_w

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
        rank_mu = np.sum(np.array([np.outer(y, y) for y in selected]), axis=0)/self._mu
        self._C = (
                (1 - self._lr_c1 - self._lr_c_mu) * self._C
                + self._lr_c1 * rank_one
                + self._lr_c_mu * rank_mu
        )

    def _objective(self, x):
        assert self._dimension > 1, 'Dimension must be greater than 1.'
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

    def plot_result(self):
        if self.results == []:
            raise Exception("Can't plot results, must run the algorithm first")
        x_axis = [iter for (iter, _) in self.results]
        y_axis = [value for (_, value) in self.results]
        plt.scatter(x_axis, y_axis)
        plt.xlabel('Timestep')
        plt.ylabel('Obj fun')
        plt.yscale('log')
        plt.grid()
        plt.show()
    
    def ecdf(self, targets: np.array):
        if self.results == []:
            raise Exception("Can't calculate ECDF, must run the algorithm first")
        ecdf = []
        for result in self.results:
            passed = 0
            for target in targets:
                if result[1] <= target:
                    passed += 1
            ecdf.append(passed/len(targets))
        missing_entries = self._stop_after - len(ecdf)
        if missing_entries > 0:
            ecdf.extend([1.]*missing_entries)
        return np.array(ecdf)
    
    def evaluate(self, iterations: int):
        ecdf_list = []
        for _ in range(iterations): #run algorithm many times, store results
            self._set_initial_params()
            self.generation_loop()
            ecdf_list.append(self.ecdf(_TARGETS))
        result_length = max([len(ecdf) for ecdf in ecdf_list])
        for ecdf in ecdf_list: #fill ecdf data so that all lists are of equal lengths
            missing_entries = result_length - len(ecdf)
            if missing_entries > 0:
                ecdf = np.append(ecdf, np.ones(missing_entries))
        ecdf_array = np.vstack(ecdf_list)
        result = np.mean(ecdf_array, axis = 0)
        print(result)
        return result



        

def felli(x: np.ndarray) -> float:
    dim = x.shape[0]
    arr = [np.power(1e6, p) for p in np.arange(0, dim) / (dim - 1)]
    return np.dot(arr, x**2)

def quadratic(x: np.ndarray) -> float:
    return np.dot(x,x)

def bent_cigar(x: np.ndarray) -> float:
    return x[0]**2 + 1e6*np.sum(x[1:]**2)

def rastrigin(x: np.ndarray) -> float:
    return np.sum(x**2 + -10*(np.cos(2*np.pi*x)) + 10)

def rosenbrock(x: np.ndarray) -> float:
    return sum([100*(x[i]**2 - x[i+1])**2 + (x[i]-1)**2 for i in range(x.shape[0]-1)])