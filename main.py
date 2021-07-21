import argparse
import Evaluator

parser = argparse.ArgumentParser(prog="CMA-ES",
                                 description='This program allows you to run CMA-ES')

parser.add_argument('-m', '--mode', type=str, default='formula',
                    help='lambda choice mode',
                    choices=['normal', 'l100', 'formula'])

parser.add_argument('-i', '--iterations', type=int, default=10,
                    help='How many algorithm runs to be averaged.')

parser.add_argument('-f', '--functions', nargs='+',
                    help='Objective functions to be used for the algorithm. '
                         'Results from all objective functions are averaged.',
                    choices=['felli', 'quadratic', 'bent', 'rastrigin', 'rosenbrock'])

parser.add_argument('-d', '--dimensions', type=int, default=10,
                    help='Number of dimensions.')

parser.add_argument('-l', '--lbd', type=int, default=100,
                    help='Population size.')

parser.add_argument('-t', '--test_case', type=str, default='all',
                    help='Which previously prepared test case to run.',
                    choices=['all'])

if __name__ == '__main__':
    args = parser.parse_args()
    if args.test_case == 'all':
        Evaluator.all_test(args.dimensions, args.iterations, args.lbd)