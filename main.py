import argparse
import Evaluator

from cmaes import CMAES

parser = argparse.ArgumentParser(prog="CMA-ES",
                                 description='This program allows you to run CMA-ES')

parser.add_argument('-m', '--mode', type=str, default='normal',
                    help='Program mode.',
                    choices=['normal', 'mean', 'mean2'])

parser.add_argument('-fr', '--frequency', type=int, default=1,
                    help='How many iteration apart should modification of the algorithm take place.'
                         'Required when mode is \'mean\', ignored otherwise')

parser.add_argument('-f', '--function', type=str, default='felli',
                    help='Objective function for the algorithm.',
                    choices=['felli', 'quadratic', 'bent', 'rastrigin', 'rosenbrock'])

parser.add_argument('-d', '--dimensions', type=int, default=10,
                    help='Number of dimensions.')


def print_mode(mode):
    print(f'Current mode is \'{mode}\'')


if __name__ == '__main__':
    args = parser.parse_args()
    Evaluator.frequencyTest(100)