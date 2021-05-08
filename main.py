import argparse

from cmaes import CMAES

parser = argparse.ArgumentParser(prog="CMA-ES",
                                 description='This program allows you to run CMA-ES')

parser.add_argument('-m', '--mode', type=str, default='normal',
                    help='Program mode',
                    choices=['normal', 'mean'])


def print_mode(mode):
    print(f'Current mode is \'{mode}\'')


if __name__ == '__main__':
    args = parser.parse_args()

    print_mode(args.mode)

    algo = CMAES()
