from cmaes import CMAES
import numpy as np
import matplotlib.pyplot as plt

_TARGETS = np.array([10 ** i for i in range(-10, 10)])


def evaluate(mode: str, repair_mode: str, dimensions: int = 10, iterations: int = 100, objectives: list = None, lambda_arg: int = None, visuals: bool = False):
    if objectives is None:
            objectives = ['quadratic', 'felli', 'bent', 'rastrigin', 'rosenbrock']

    ecdf_list = []
    evals_per_gen = None
    print("Starting evaluation...")
    lambda_prompt = str(lambda_arg) if lambda_arg is not None else "default"
    print(f"mode: {mode}; dimensions: {dimensions}; iterations: {iterations}; population: {lambda_prompt}; repair: {repair_mode}")
    for objective in objectives:
        print("    Currently running:", objective)
        for _ in range(iterations):
            algo = CMAES(objective, dimensions, mode, repair_mode, lambda_arg, visuals)
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

def all_test(dimensions: int, iterations: int, lbd: int):
    runs = [
        ('normal', None, False),
        ('constrained', 'reflection', False),
        ('constrained', 'projection', False),
        ('constrained', 'resampling', False)
    ]
    ecdfs = []
    for mode, rmode, v in runs:
        ecdf = evaluate(mode, rmode, dimensions, iterations, None, lbd, False)
        ecdfs.append((ecdf[0], ecdf[1], str(rmode)))
    for ecdf in ecdfs:
        plt.scatter(ecdf[0], ecdf[1], label=ecdf[2])
    plt.legend()
    plt.grid()
    plt.show()