from cmaes import CMAES
import numpy as np
import matplotlib.pyplot as plt

_TARGETS = np.array([10 ** i for i in range(-10, 10)])


def evaluate(mode: str, dimensions: int = 10, iterations: int = 100, lambda_arg: int = 100, objectives: list = None):
    if objectives is None:
        objectives = ['quadratic', 'felli', 'bent']

    ecdf_list = []
    print("Starting evaluation...")
    print(f"mode: {mode}; dimensions: {dimensions}; iterations: {iterations}; "
          f"population: {lambda_arg}")
    for objective in objectives:
        print("    Currently running:", objective)
        for i in range(iterations):
            algo = CMAES(objective, dimensions, mode, lambda_arg)
            algo.generation_loop()
            ecdf_list.append(algo.ecdf(_TARGETS))

    return _get_ecdf_data(ecdf_list)


def _get_ecdf_data(ecdf_list: list):
    max_length = max([len(ecdf) for ecdf in ecdf_list])

    for ecdf in ecdf_list:  # fill ecdf data with 1s so that all lists are of equal lengths
        missing_entries = max_length - len(ecdf)
        if missing_entries > 0:
            ecdf.extend([1.] * missing_entries)

    ecdf_result = []
    for i in range(max_length):
        ecdf_result.append(sum([ecdf[i] for ecdf in ecdf_list]) / len(ecdf_list))
    return ecdf_result



def modifications_test(dimensions: int, iterations: int, lambda_arg: int):
    ecdf_normal = evaluate('normal', dimensions, iterations, lambda_arg, None)
    ecdf_mean_all = evaluate('mean_all', dimensions, iterations, lambda_arg, 1)
    ecdf_mean_selected = evaluate('mean_selected', dimensions, iterations, lambda_arg, 1)
    plt.plot(ecdf_normal, label='normal')
    plt.plot(ecdf_mean_all, label='mean_all 1')
    plt.plot(ecdf_mean_selected, label='mean_selected 1')
    plt.legend()
    plt.show()


def all_test(dimensions: int, iterations: int, lambda_arg: int):
    ecdf_mode_1 = evaluate('mode_1', dimensions, iterations, lambda_arg)
    ecdf_mode_2 = evaluate('mode_2', dimensions, iterations, lambda_arg)
    ecdf_mode_3 = evaluate('mode_3', dimensions, iterations, lambda_arg)
    plt.plot(ecdf_mode_1, label='mode_1')
    plt.plot(ecdf_mode_2, label='mode_2')
    plt.plot(ecdf_mode_3, label='mode_3')
    plt.legend()
    plt.show()

def one_test():
    ecdf = evaluate('the only', 2, 100, 100, objectives=['quadratic'])
    plt.plot(ecdf, label='ecdf')
    plt.legend()
    plt.show()