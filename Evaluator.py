from cmaes import CMAES
import numpy as np
import matplotlib.pyplot as plt
from sys import stdout

_TARGETS = np.array([10 ** i for i in range(-20, 2)])
# _TARGETS = np.array([10 ** (i/8.0) for i in range(-16, 17)])

def evaluate(repair_mode: str, dimensions: int = 10, iterations: int = 10, objectives: list = None, lambda_arg: int = None, visual: bool = False):
    # return formatted ecdf data and formatted sigma data

    if objectives is None:
            # objectives = ['quadratic', 'felli', 'bent', 'rastrigin', 'rosenbrock', 'ackley']
            objectives = ['felli']
    ecdf_list = []
    sigmas_list = []
    evals_per_gen = None
    print("Starting evaluation...")
    lambda_prompt = str(lambda_arg) if lambda_arg is not None else "default"
    print(f"dimensions: {dimensions}; iterations: {iterations}; population: {lambda_prompt}; repair: {repair_mode}")
    for objective in objectives:
        print("    Currently running:", objective)
        for iteration in range(iterations):
            stdout.write(f"\rIteration: {1+iteration} / {iterations}")
            stdout.flush()
            algo = CMAES(objective, dimensions, repair_mode, lambda_arg, visual)
            algo.generation_loop()
            if evals_per_gen == None:
                evals_per_gen = algo.evals_per_iteration()
            else:
                assert evals_per_gen == algo.evals_per_iteration(), "Lambda different for same settings"
            ecdf_list.append(algo.ecdf(_TARGETS))
            sigmas_list.append(algo.sigma_history())
        print()

    formatted_ecdfs = _format_ecdfs(ecdf_list, evals_per_gen)
    formatted_sigmas = _format_sigmas(sigmas_list, evals_per_gen)
    return (formatted_ecdfs, formatted_sigmas)


def _format_ecdfs(ecdf_list: list, evals_per_gen: int):
    max_length = max([len(ecdf) for ecdf in ecdf_list])

    for ecdf in ecdf_list:  # fill ecdf data with 1s so that all lists are of equal lengths
        missing_entries = max_length - len(ecdf)
        if missing_entries > 0:
            ecdf.extend([1.] * missing_entries)

    y_axis = []
    for i in range(max_length):
        y_axis.append(sum([ecdf[i] for ecdf in ecdf_list]) / len(ecdf_list))
    x_axis = [x*evals_per_gen for x in range(max_length)]
    return x_axis, y_axis

def _format_sigmas(sigmas_list: list, evals_per_gen: int):
    max_length = len(sigmas_list[0])
    for i in sigmas_list:
        assert len(i) == max_length, "Runs are of different length, cannot take average of sigma"
    y_axis = []
    for i in range(max_length):
        y_axis.append(sum([sigmas[i] for sigmas in sigmas_list if sigmas[i] is not None]) / len(sigmas_list))
    x_axis = [x*evals_per_gen for x in range(max_length)]
    return x_axis, y_axis

def all_test(dimensions: int, iterations: int, lbd: int, visual: bool):
    runsc = [
        (None, False),
        ('reflection', True),
        ('projection', True),
        ('resampling', True)
    ]
    ecdfs = []
    sigmas = []
    for rmode, v in runsc:
        ecdf, sigma = evaluate(rmode, dimensions, iterations, None, lbd, visual and v)
        ecdfs.append((ecdf[0], ecdf[1], str(rmode)))
        sigmas.append((sigma[0], sigma[1], str(rmode)))

    lambda_prompt = str(lbd) if lbd is not None else "default"
    title = f"dimensions: {dimensions}; iterations: {iterations}; population: {lambda_prompt}"
    ecdf_ax = plt.subplot(211)
    plt.title(title, fontsize=18)
    plt.setp(ecdf_ax.get_xticklabels(), fontsize = 8)
    for ecdf in ecdfs:
        plt.plot(ecdf[0], ecdf[1], label=ecdf[2])
    plt.legend()
    
    sigma_ax = plt.subplot(212, sharex=ecdf_ax)
    plt.setp(ecdf_ax.get_xticklabels(), visible = False)
    for sigma in sigmas:
        plt.plot(sigma[0], sigma[1], label=sigma[2])
    plt.yscale("log")
    # ecdf_ax.ylabel("ECDF_curves", loc="left")
    # ecdf_ax.set_ylim(0,1)
    # sigma_ax.legend()
    # sigma_ax.xlabel("sigma_curves", loc="left")
    plt.show()
