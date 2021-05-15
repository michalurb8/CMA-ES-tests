import numpy as np
import math
import matplotlib.pyplot as plt


class CMAES:

    def __init__(self, objective_function: str, dimensions: int):
        self._fitness = 'quadratic'
        self._dimension = 2
        # self._fitness = objective_function
        # self._dimension = dimensions
        # Initial point
        self._xmean = np.random.rand(self._dimension)[:, np.newaxis]
        self._zmean = None
        # Step size
        self._sigma = 0.05
        self._stop_value = 1e-10
        self._stop_after = 1e3 * self._dimension ** 2

        print(f'Dimension: {self._dimension}\n'
              f'_fitness: {self._fitness}\n'
              f'Xmean: {self._xmean}\n'
              f'Sigma: {self._sigma}\n'
              f'stop_value: {self._stop_value}\n'
              f'stop_after: {self._stop_after}\n')

        # Set up selection
        # Population size
        self._lambda = int(4 + np.floor(3 * np.log(self._dimension)))
        # Number of parents/points for recombination
        self._mu = self._lambda // 2
        # Recombination weights
        self._weights = np.log(self._mu + 0.5) - np.log(np.arange(1, self._mu + 1))
        # self._weights = np.ones(self._mu)
        # Normalize weights
        self._weights = self._weights / sum(self._weights)
        # Variance effective number of parents
        self._mu_effective = sum(self._weights) ** 2 / sum(self._weights ** 2)
        self._weights = self._weights[:, np.newaxis]

        print(f'Lambda: {self._lambda}\n'
              f'Mu: {self._mu}\n'
              f'weights: {self._weights}\n'
              f'_mu_effective: {self._mu_effective}\n')

        # Set up adaptation
        # Time constants for cumulation
        self._time_c = self._get_time_constant()
        self._time_sigma = (self._mu_effective + 2) / (self._dimension + self._mu_effective + 5)
        # Learning rates for rank-one and rank-mu update
        self._lr_c = self._get_c_learning_rate()
        self._lr_mu = self._get_mu_learning_rate()
        self._damping = self._get_sigma_damping()

        # print(f'_cc: {self._time_c}\n'
        #       f'_cs: {self._time_sigma}\n'
        #       f'_lr_c: {self._lr_c}\n'
        #       f'_lr_mu: {self._lr_mu}\n'
        #       f'_damp: {self._damping}\n')

        # Initialize internal parameters
        # Evolution paths
        self._path_c = np.zeros((self._dimension, 1))
        self._path_sigma = np.zeros((self._dimension, 1))
        # B defines coordinate system
        self._B = np.eye(self._dimension)
        # D defines scaling (diagonal matrix)
        self._D = np.eye(self._dimension)
        # Covariance matrix
        self._C = np.eye(self._dimension)
        # Parameter for B and D update timing
        self._eigen_eval = 0
        self._chi = np.sqrt(self._dimension) * (1 - 1 / (4 * self._dimension) + 1 / (21 * self._dimension ** 2))

        # print(f'_pc: {self._path_c}\n'
        #       f'_ps: {self._path_sigma}\n'
        #       f'_B: {self._B}\n'
        #       f'_D: {self._D}\n'
        #       f'_C: {self._C}\n'
        #       f'_eigen_eval: {self._eigen_eval}\n'
        #       f'_chi: {self._chi}\n')

    def _get_time_constant(self):
        return ((4 + self._mu_effective / self._dimension) /
                (4 + self._dimension + 2 * self._mu_effective / self._dimension))

    def _get_c_learning_rate(self):
        return 2 / ((self._dimension + 1.3) ** 2 + self._mu_effective)

    def _get_mu_learning_rate(self):
        return (2 * (self._mu_effective - 2 + 1 / self._mu_effective) /
                ((self._dimension + 2) ** 2 + 2 * self._mu_effective / 2))

    def _get_sigma_damping(self):
        return 1 + 2 * max(0, np.sqrt((self._mu_effective - 1) / (self._dimension + 1)) - 1) + self._time_sigma

    def generation_loop(self):
        count_it = 0
        arx, indices, arfitness = None, None, None
        results = []
        while count_it < self._stop_after:
            arfitness, arx, arz = self._generate_offspring()
            count_it += self._lambda

            # print(f'arz: {arz}\n')
            # print(f'arz.shape: {arz.shape}\n')
            # print(f'arx: {arx}\n')
            # print(f'arx.shape: {arx.shape}\n')
            # print(f'arfit: {arfitness}')

            indices = arfitness.argsort(axis=0).flatten()
            arfitness = arfitness[indices]

            self._recalculate_mean(arx, arz, indices)

            # print(f'self._xmean: {self._xmean}\n')
            # print(f'self._zmean: {self._zmean}\n')

            # Update evolution paths for C and sigma
            ps_delta = (np.sqrt(self._time_sigma * (2 - self._time_sigma) * self._mu_effective) *
                        np.matmul(self._B, self._zmean))
            self._path_sigma = (1 - self._time_sigma) * self._path_sigma + ps_delta

            # print(f'self._ps: {self._ps}\n')

            # Heaviside function
            h_sigma = int(np.linalg.norm(self._path_sigma) /
                          np.sqrt(1 - np.power(1 - self._time_sigma, 2 * count_it / self._lambda)) / self._chi
                          < 1.4 + 2 / (self._dimension + 1))

            # print(f'h_sigma: {h_sigma}\n')

            self._path_c = ((1 - self._time_c) * self._path_c +
                            h_sigma * np.sqrt(self._time_c * (2 - self._time_c) * self._mu_effective) *
                            (np.matmul(np.matmul(self._B, self._D), self._zmean)))

            # print(f'self._pc: {self._pc}\n')

            # Update covariance matrix
            mut_mat = np.matmul(np.matmul(self._B, self._D), arz[:, indices[range(self._mu)]])
            self._C = ((1 - self._lr_c - self._lr_mu) * self._C +
                       self._lr_c * (np.matmul(self._path_c, np.transpose(self._path_c))
                                     + (1 - h_sigma) * self._time_c * (2 - self._time_c) * self._C) +
                       self._lr_mu * np.matmul(np.matmul(mut_mat, np.diag(self._weights.flatten())),
                                               np.transpose(mut_mat)))

            # print(f'self._C: {self._C}\n')
            # print(f'self._C.shape: {self._C.shape}\n')

            self._sigma = self._sigma * np.exp(
                (self._time_sigma / self._damping) * (np.linalg.norm(self._path_sigma) / self._chi - 1))

            # print(f'self._sigma: {self._sigma}\n')

            if count_it - self._eigen_eval > self._lambda / (self._lr_c + self._lr_mu) / self._dimension / 10:
                self._eigen_eval = count_it
                self._C = np.triu(self._C) + np.transpose(np.triu(self._C, 1))
                self._D, self._B = np.linalg.eig(self._C)
                # print(f'self._D before: {self._D}\n')
                self._D = np.diag(np.sqrt(np.abs(self._D)))

            # print(f'self._eigen_eval: {self._eigen_eval}\n')
            # print(f'self._C: {self._C}\n')
            # print(f'self._B: {self._B}\n')
            # print(f'self._D: {self._D}\n')

            if arfitness[1] <= self._stop_value:
                print('arfitness[1] <= self._stop_value')
                break

            if arfitness[1] == arfitness[math.ceil(0.7 * self._lambda)]:
                self._sigma = self._sigma * np.exp(0.2 + self._time_sigma / self._damping)
                print('warning: flat fitness, consider reformulating the objective')

            print(f'Evaluation iteration {count_it} - objective value: {arfitness[1]}')
            results.append(arfitness[1])
            if np.abs(arfitness[1]) > 1e16:
                print('This is getting out of hand.')
                break

        print(f'Evaluation iteration {count_it} - objective value: {arfitness[1]}')
        print(arx[:, indices[1]])
        x_axis = np.arange(len(results)) * 10
        plt.plot(x_axis, results)
        plt.xlabel('Timestep')
        plt.ylabel('Obj fun')
        plt.grid()
        plt.show()

    def _recalculate_mean(self, arx, arz, indices):
        self._xmean = np.matmul(arx[:, indices[range(self._mu)]], self._weights)
        self._zmean = np.matmul(arz[:, indices[range(self._mu)]], self._weights)

    def _generate_offspring(self):
        arz = np.array(generate_random_matrix(self._dimension, self._lambda))
        arx = np.array(self._get_mutation(arz))
        arfitness = np.array([self._objective(x) for x in arx.T])
        return arfitness, arx, arz

    def _get_mutation(self, random_matrix):
        result = self._sigma * np.matmul(np.matmul(self._B, self._D), random_matrix)
        for k in range(self._lambda):
            result[:, k] += self._xmean.flatten()
        return result

    def _objective(self, x):
        if self._fitness == 'felli':
            assert self._dimension > 1, 'Dimension must be greater than 1.'
            return felli(x)
        elif self._fitness == 'quadratic':
            # assert self._dimension == 1, 'Invalid dimension for quadratic function.'
            return quadratic(x)
        raise Exception('Invalid objective function chosen')


def generate_random_matrix(x, y):
    return [np.random.rand(y) for _ in range(x)]


def felli(x: np.ndarray):
    dim = x.shape[0]
    arr = [np.power(1e6, p) for p in np.subtract(np.arange(1, dim + 1), 1) / (dim - 1)]
    return np.matmul(arr, x ** 2)


def quadratic(arr: np.ndarray):
    # return [x ** 2 for x in arr]
    return arr[0] ** 2 + arr[1] ** 2
    # return [-3 * x ** 2 + 4.5 * x + 7 for x in arr]
