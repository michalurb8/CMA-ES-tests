import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

_EPS = 1e-50
_POINT_MAX = 1e50
_SIGMA_MAX = 1e50

_DELAY = 1

infp = float('inf')
infn = float('-inf')
_RESAMPLING_LIMIT = 10

class CMAES:
    """
    Parameters
    ----------
    objective_function: str
        Chosen from: quadratic, felli, bent, rastrigin, rosenbrock, ackley
    dimensions: int
        Objective function dimensionality.
    repair_mode: str
        Bound constraint repair method. Chosen from: None, projection, reflection, resampling.
    lambda_arg: int
        Population count. Must be > 3, if set to None, default value will be computed.
    stop_after: int
        How many iterations are to be run
    visuals: bool
        If True, every algorithm generation will be visualised (only 2 first dimensions)
    move_delta: bool
        If True, delta (y_w) will be change by the sum of repair vectors
    """
    def __init__(self, objective_function: str, dimensions: int, repair_mode: str, lambda_arg: int = None, stop_after: int = 50, visuals: bool = False, move_delta: bool = False):
        assert dimensions > 0, "Number of dimensions must be greater than 0"
        self._dimension = dimensions
        self._fitness = objective_function
        self._repair_mode = repair_mode
        self._stop_after = stop_after
        self._visuals = visuals
        self._move_delta = move_delta

        assert self._repair_mode in [None, 'projection', 'reflection', 'resampling'], 'Incorrect repair mode'

        # Set bounds:
        self._bounds = [(-0.1,100) for _ in range(self._dimension)]

        # Initial point
        self._xmean = 10 + 80 * np.random.uniform(size = self._dimension)
        # Step size
        self._sigma = 1

        # Population size
        if lambda_arg == None:
            self._lambda = 4 * self._dimension # default population size
        else:
            assert lambda_arg > 3, "Population size must be greater than 3"
            self._lambda = lambda_arg

        # Number of parents/points to be selected
        self._mu = self._lambda // 2

        # Learning rates for rank-one and rank-mu update
        self._lr_c1 = 2 / ((self._dimension + 1.3) ** 2 + self._mu)
        self._lr_c_mu = (2 * (self._mu - 2 + 1 / self._mu) /
                         ((self._dimension + 2) ** 2 + self._mu))
        self._lr_c_mu = min(1-self._lr_c1, self._lr_c_mu)

        # Time constants for cumulation for step-size control
        self._time_sigma = (self._mu + 2) / (self._dimension + self._mu + 5)
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

        # Store current generation number
        self._generation = 0

        # Store important values at each generation
        self._results = []
        self._sigma_history = []
        self._eigen_history = np.zeros((self._stop_after, self._dimension))
        self._mean_history = []
        self._repair_history = []

        # Store best found value so far for ECDF calculation
        self._best_value = infp

        # Run the algorithm immediately
        self._generation_loop()

    def _generation_loop(self):
        assert self._results == [], "One algorithm instance can only run once."
        for gen_count in range(self._stop_after):
            self._B, self._D = self._eigen_decomposition()

            self._sigma_history.append(self._sigma)
            self._eigen_history[self._generation, :] = np.multiply(self._D, np.ones(self._dimension))

            solutions = [] # this is a list of tuples (x, y , value)
            repair_count = 0
            for _ in range(self._lambda):
                x = self._sample_solution() # x is the original point
                y = self._repair(x, self._bounds, self._repair_mode) # y is the repaired point
                if not np.all(x == y):
                    repair_count += 1

                value = self._objective(x)
                self._best_value = min(value, self._best_value)
                solutions.append((x, y, value))
            # Update algorithm parameters.
            assert len(solutions) == self._lambda, "There must be exactly lambda points generated"
            self._repair_history.append(repair_count)
            self._update(solutions, repair_count)
            self._results.append((gen_count, self._best_value))

    def _sample_solution(self) -> np.ndarray:
        std = np.random.standard_normal(self._dimension)
        return self._xmean + self._sigma * np.matmul(np.matmul(self._B, np.diag(self._D)), std)

    def _eigen_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        return B, D

    def _update(self, solutions: List[Tuple[np.ndarray, np.ndarray, float]], repair_count: int) -> None:
        assert len(solutions) == self._lambda, "Must evaluate solutions with length equal to population size."
        for s in solutions:
            assert np.all(
                np.abs(s[0]) < _POINT_MAX
            ), f"Absolute value of all generated points must be less than {_POINT_MAX} to avoid overflow errors."
            assert np.all(
                np.abs(s[1]) < _POINT_MAX
            ), f"Absolute value of all repaired points must be less than {_POINT_MAX} to avoid overflow errors."

        self._generation += 1
        solutions.sort(key=lambda solution: solution[-1]) #sort population by function value

        # ~ N(m, sigma^2 C)
        originals = np.array([s[0] for s in solutions])
        population = np.array([s[1] for s in solutions])
        # ~ N(0, C)
        y_k = (population - self._xmean) / (self._sigma + _EPS)

        # Selection
        selected = y_k[: self._mu]
        y_w = np.mean(selected, axis=0) # cumulated delta vector

        # Delta correction step
        if self._move_delta:
            # Delta 1: difference between generated points mean and repaired points mean:
            delta1 = (np.mean(originals, axis=0) - np.mean(population, axis=0)) / (self._sigma + _EPS)
            # Delta 2: difference between selected generated points mean and selected repaired points mean:
            delta2 = (np.mean(originals[:self._mu], axis=0) - np.mean(population[:self._mu], axis=0)) / (self._sigma + _EPS)

            alpha = 0.5 * repair_count/self._lambda
            delta1_scaled = _resize(delta1, y_w, alpha)
            delta2_scaled = _resize(delta2, y_w, alpha)

            correction = delta2_scaled
            y_w += correction

        self._xmean += self._sigma * y_w

        self._mean_history.append(self._objective(self._xmean))

        if self._visuals == True:
            title = "gen " + str(self._generation)
            title += ", repair_mode: " + str(self._repair_mode)
            title += ", lambda: " + str(self._lambda)
            title += ", dim: " + str(self._dimension)
            title += ", correction: " + str(self._move_delta)
            plt.title(title)

            # plt.axis('equal')

            plt.axvline(0, linewidth=4, c='black')
            plt.axhline(0, linewidth=4, c='black')
            plt.axvline(self._bounds[-1][0], linewidth=2, c='red')
            plt.axvline(self._bounds[-2][1], linewidth=2, c='red')
            plt.axhline(self._bounds[-1][0], linewidth=2, c='red')
            plt.axhline(self._bounds[-2][1], linewidth=2, c='red')
            x1 = [point[-1] for point in population]
            x2 = [point[-2] for point in population]
            plt.scatter(x1, x2, s=50)
            x1 = [point[-1] for point in population[:self._mu]]
            x2 = [point[-2] for point in population[:self._mu]]
            plt.scatter(x1, x2, s=15)
            plt.scatter(self._xmean[-1], self._xmean[-2], s=100, c='black')
            plt.grid()
            zoom_out = 1.3
            max1 = zoom_out*max([abs(point[-1]) for point in population])
            max2 = zoom_out*max([abs(point[-2]) for point in population])
            plt.xlim(-max1, max1)
            plt.ylim(-max2, max2)
            plt.pause(_DELAY)
            plt.clf()
            plt.cla()

        # Step-size control
        # C^(-1/2) = B D^(-1) B^T
        C_2 = np.matmul(np.matmul(self._B, np.diag(1 / (self._D + _EPS))), self._B.T)

        _path_sigma_delta = (np.sqrt(self._time_sigma * (2 - self._time_sigma) * self._mu) *
                        np.matmul(C_2, y_w))
        self._path_sigma = (1 - self._time_sigma) * self._path_sigma + _path_sigma_delta

        self._sigma *= np.exp( (self._time_sigma / (self._damping + _EPS)) *
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
    
    def _objective(self, x):
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
        elif self._fitness == 'ackley':
            return ackley(x)
        raise Exception('Invalid objective function chosen: ', str(self._fitness))

    def sigma_history(self) -> List[float]:
        return self._sigma_history

    def diff_history(self) -> List[float]:
        diffs = [1]
        for i in range(len(self._sigma_history) - 1):
            diffs.append(self._sigma_history[i+1] / self._sigma_history[i])
        return diffs

    def cond_history(self) -> List[float]:
        greatest = self._eigen_history[:,-1]
        smallest = self._eigen_history[:,0]
        result = greatest/(smallest+_EPS)
        return result
        
    def eigen_history(self) -> np.ndarray:
        return self._eigen_history

    def mean_history(self) -> List[float]:
        return self._mean_history

    def ecdf_history(self, targets: np.array) -> List[float]:
        ecdf = []
        for result in self._results:
            passed = 0
            for target in targets:
                if result[1] <= target:
                    passed += 1
            ecdf.append(passed / len(targets))
        return ecdf
    
    def repair_history(self) -> List[float]:
        return self._repair_history

    def evals_per_iteration(self) -> int:
        return self._lambda

    def _repair(self, x: np.array, bounds: List[Tuple[float, float]], repair_mode: str) -> np.array:
        """
        Parameters
        ----------
        x: numpy array
            Represents a single point in search space
        bounds: List of dim Tuples of two floats
            Describe lower and upper bound in each dimension
        repair_mode: str
            Bound constraint repair method. Chosen from: None, projection, reflection, resampling.
        ----------
        This method repairs x according to repair_mode and bounds.
        x is replaced with the repaired value.
        Repair vector is returned, equal to x - original_x.
        """
        assert self._dimension == len(bounds), "Constraint number and dimensionality do not match"
        repaired = np.copy(x)
        if repair_mode == None:
            pass
        elif self._check_point(x):
            pass
        elif repair_mode == 'reflection':
            for i in range(len(x)):
                while repaired[i] < bounds[i][0] or repaired[i] > bounds[i][1]:
                    if repaired[i] < bounds[i][0]:
                        repaired[i] = 2 * bounds[i][0] - repaired[i]
                    if repaired[i] > bounds[i][1]:
                        repaired[i] = 2 * bounds[i][1] - repaired[i]
        elif repair_mode == 'projection':
            for i in range(len(repaired)):
                if repaired[i] < bounds[i][0]:
                    repaired[i] = bounds[i][0]
                elif repaired[i] > bounds[i][1]:
                    repaired[i] = bounds[i][1]
        elif repair_mode == 'resampling':
            for _ in range(_RESAMPLING_LIMIT):
                new = self._sample_solution()
                for i in range(len(repaired)):
                    repaired[i] = new[i]
                if self._check_point(repaired):
                    break
            repaired = self._repair(repaired, bounds, 'projection')
        else:
            raise Exception("Incorrect repair mode")
        return repaired
    
    def _check_point(self, x: np.array):
        for i in range(len(x)):
            if x[i] < self._bounds[i][0] or x[i] > self._bounds[i][1]:
                return False
        return True

def felli(x: np.ndarray) -> float:
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

def _resize(vec1: np.array, vec2: np.array, const: float) -> None:
    #returns vec1 rescaled to const*lenght of vec2
    vec1 = vec1 / (np.linalg.norm(vec1) + _EPS)
    scale = const * np.linalg.norm(vec2)
    vec1 *= scale
    return vec1