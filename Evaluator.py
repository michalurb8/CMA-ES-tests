from cmaes import CMAES
import numpy as np
import matplotlib.pyplot as plt

_TARGETS = np.array([10 ** i for i in range(-10, 10)])


def evaluate(mode: str, dimensions: int = 10, iterations: int = 100, lambda_arg: int = 100, objectives: list = None):
    if objectives is None:
        objectives = ['quadratic', 'felli', 'bent']

    ecdf_list = []
    evals_per_gen = None
    print("Starting evaluation...")
    print(f"mode: {mode}; dimensions: {dimensions}; iterations: {iterations}; "
          f"population: {lambda_arg}")
    for objective in objectives:
        print("    Currently running:", objective)
        for _ in range(iterations):
            algo = CMAES(objective, dimensions, mode, lambda_arg)
            algo.generation_loop()
            single_ecdf, e = algo.ecdf(_TARGETS)
            ecdf_list.append(single_ecdf)
            if evals_per_gen == None:
                evals_per_gen = e
            else:
                assert evals_per_gen == e, "Lambda different for same settings"

    return _get_ecdf_data(ecdf_list, evals_per_gen)


def _get_ecdf_data(ecdf_list: list, evals_per_gen: int):
    max_length = max([len(ecdf) for ecdf in ecdf_list])

    for ecdf in ecdf_list:  # fill ecdf data with 1s so that all lists are of equal lengths
        missing_entries = max_length - len(ecdf)
        if missing_entries > 0:
            ecdf.extend([1.] * missing_entries)

    ecdf_result = []
    for i in range(max_length):
        ecdf_result.append(sum([ecdf[i] for ecdf in ecdf_list]) / len(ecdf_list))
    xaxis = [x*evals_per_gen for x in range(max_length)]
    return xaxis, ecdf_result

def all_test(dimensions: int, iterations: int, lambda_arg: int):
    for l in [3,4,5,6,7,8,9,10]:
        ecdf = evaluate('normal', dimensions, iterations, l)
        plt.scatter(ecdf[0], ecdf[1], label=str(l))
    plt.legend()
    plt.grid()
    plt.show()