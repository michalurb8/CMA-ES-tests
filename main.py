import argparse
import numpy as np

from cmaes import CMAES

parser = argparse.ArgumentParser(prog="CMA-ES",
                                 description='This program allows you to run CMA-ES')

parser.add_argument('-m', '--mode', type=str, default='normal',
                    help='Program mode.',
                    choices=['normal', 'mean'])

parser.add_argument('-f', '--function', type=str, default='felli',
                    help='Objective function for the algorithm.',
                    choices=['felli', 'quadratic'])

parser.add_argument('-d', '--dimensions', type=int, default=10,
                    help='Number of dimensions.')


def print_mode(mode):
    print(f'Current mode is \'{mode}\'')


if __name__ == '__main__':
    args = parser.parse_args()

    print_mode(args.mode)

    algo = CMAES(args.function, args.dimensions)
    algo.generation_loop()


