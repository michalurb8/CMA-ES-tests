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

parser.add_argument('-s', '--stop', type=int, default=100,
                    help='How many iterations to take average of.')

parser.add_argument('-v', '--vis', default=False,
                    help='Turn on visualisation.', action='store_true')

parser.add_argument('-r', '--repair', type=str, default=None,
                    help='Repair method')

if __name__ == '__main__':
    args = parser.parse_args()
    Evaluator.run_test(args.dimensions, args.iterations, args.lbd, args.stop, args.vis, args.repair)
