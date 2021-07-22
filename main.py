import argparse
import Evaluator

parser = argparse.ArgumentParser(prog="CMA-ES",
                                 description='This program allows you to run CMA-ES')

parser.add_argument('-i', '--iterations', type=int, default=10,
                    help='How many algorithm runs to be averaged.')

parser.add_argument('-d', '--dimensions', type=int, default=10,
                    help='Number of dimensions.')

parser.add_argument('-l', '--lbd', type=int, default=None,
                    help='Population size.')

if __name__ == '__main__':
    args = parser.parse_args()
    Evaluator.all_test(args.dimensions, args.iterations, args.lbd)