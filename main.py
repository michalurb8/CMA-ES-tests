import argparse
import Evaluator

parser = argparse.ArgumentParser(prog="CMA-ES",
                                 description='This program allows you to run CMA-ES')

parser.add_argument('-m', '--mode', type=str, default='normal',
                    help='Program mode. Normal will run default CMA-ES implementation, while \'mean_all\' and '
                         '\'mean_selected\' will run algorithm with modification adding a mean point to the '
                         'population, calculated respectively either from whole population all best percentage of it.',
                    choices=['normal', 'mean_all', 'mean_selected'])

parser.add_argument('-i', '--iterations', type=int, default=100,
                    help='How many times the algorithm should be run.')

parser.add_argument('-fr', '--frequency', type=int, default=1,
                    help='How many iteration apart should modification of the algorithm take place.'
                         'Ignored if \'normal\' mode is selected')

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
                    choices=['selected_frequency', 'all_frequency', 'modifications', 'all', 'custom', 'only'])

if __name__ == '__main__':
    args = parser.parse_args()
    if args.test_case == 'selected_frequency':
        Evaluator.selected_frequency_test(args.dimensions, args.iterations, args.lbd)
    elif args.test_case == 'all_frequency':
        Evaluator.all_frequency_test(args.dimensions, args.iterations, args.lbd)
    elif args.test_case == 'modifications':
        Evaluator.modifications_test(args.dimensions, args.iterations, args.lbd)
    elif args.test_case == 'all':
        Evaluator.all_test(args.dimensions, args.iterations, args.lbd)
    elif args.test_case == 'only':
        Evaluator.one_test()
    elif args.test_case == 'custom':
        Evaluator.custom_test(dimensions=args.dimensions, objectives=args.functions,
                              iterations=args.iterations, mode=args.mode, lambda_arg=args.lbd)
