from cmaes import CMAES
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sys import stdout

_TARGETS = np.array([10 ** i for i in range(-10, 10)])

def evaluate(repair_mode: str, dimensions: int = 10, iterations: int = 10, objectives: List = None, lambda_arg: int = None, visual: bool = False):
    # reapetedly run the algorithm, return results: ECDF values, sigma values, sigma difference values
    """
    evaluate() runs the algorithm multiple times (exactly 'iteration' times).
    Data about ecdf, sigma, condition number etc. is collected.
    Then, data is averaged across different iterations and returned as a tuple of lists, each list ready to be plotted.
    Parameters
    ----------
    repair_mode : str
        Bound constraint repair method. Chosen from: None, projection, reflection, resampling.
    dimensions : int
        Objective function dimensionality.
    iterations : int
        Number of algorithm runs to take an average of.
    objectives: List[str]
        List of objective functions. For each function, the algorithm will run 'iteration' times,
        then an average of all runs for all objective functions will be computed.
        Chosen from: quadratic, felli, bent, rastrigin, rosenbrock, ackley
    lambda_arg : int
        Population count. Must be > 3, if set to None, default value will be computed.
    visual: bool
        If True, every algorithm generation will be visualised (only 2 first dimensions)
    """
    if objectives is None:
            objectives = ['felli']
    ecdfs_list = []
    sigmas_list = []
    diffs_list = []
    eigens_list = []
    cond_list = []
    evals_per_gen = None
    print("Starting evaluation...")
    lambda_prompt = str(lambda_arg) if lambda_arg is not None else "default"
    print(f"dimensions: {dimensions}; iterations: {iterations}; population: {lambda_prompt}; repair: {repair_mode}")
    for objective in objectives:
        print("    Currently running:", objective)
        for iteration in range(iterations):
            stdout.write(f"\rIteration: {1+iteration} / {iterations}")
            stdout.flush()
            algo = CMAES(objective, dimensions, repair_mode, lambda_arg, visual) # algorithm runs here
            algo.generation_loop()
            if evals_per_gen == None:
                evals_per_gen = algo.evals_per_iteration()
            else:
                assert evals_per_gen == algo.evals_per_iteration(), "Lambda different for same settings"
            ecdfs_list.append(algo.ecdf(_TARGETS))
            sigmas_list.append(algo.sigma_history())
            diffs_list.append(algo.diff_history())
            eigens_list.append(algo.eigen_history())
            cond_list.append(algo.cond_history())
        print()

    formatted_ecdfs = _format_ecdfs(ecdfs_list, evals_per_gen)
    formatted_sigmas = _format_sigmas(sigmas_list, evals_per_gen)
    formatted_diffs = _format_sigma_differences(diffs_list, evals_per_gen)
    formatted_eigens = _format_eigenvalues(eigens_list, evals_per_gen)
    formatted_conds = _format_condition_numbers(cond_list, evals_per_gen)
    return (formatted_ecdfs, formatted_sigmas, formatted_diffs, formatted_eigens, formatted_conds)


def _format_ecdfs(ecdf_list: List, evals_per_gen: int) -> Tuple:
    """
    _format_ecdfs() takes data collected from multiple algorithm runs.
    Then, an average is computed and horizontal axis scaling is applied.
    The return value is a tuple of two list, each corresponds to a plot axis, ready to be plotted.
    """
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

def _format_sigmas(sigmas_list: List, evals_per_gen: int) -> Tuple:
    """
    _format_sigmas() takes data collected from multiple algorithm runs.
    Then, an average and horizontal axis scaling are applied.
    The return value is a tuple of two list, each corresponds to a plot axis, ready to be plotted.
    """
    max_length = len(sigmas_list[0])
    for i in sigmas_list:
        assert len(i) == max_length, "Runs are of different length, cannot take average of sigma"
    y_axis = []
    for i in range(max_length):
        y_axis.append(sum([sigmas[i] for sigmas in sigmas_list if sigmas[i] is not None]) / len(sigmas_list))
    x_axis = [x*evals_per_gen for x in range(max_length)]
    return x_axis, y_axis

def _format_sigma_differences(diff_list: List, evals_per_gen: int) -> Tuple:
    """
    _format_diffs() takes difference data collected from multiple algorithm runs.
    Then, an average is computed and horizontal axis scaling is applied.
    The return value is a tuple of two list, each corresponds to a plot axis, ready to be plotted.
    """
    max_length = len(diff_list[0])
    for i in diff_list:
        assert len(i) == max_length, "Runs are of different length, cannot take average of sigma differences"
    y_axis = []
    for i in range(max_length):
        y_axis.append(sum([diffs[i] for diffs in diff_list if diffs[i] is not None]) / len(diff_list))
    x_axis = [x*evals_per_gen for x in range(max_length)]
    return x_axis, y_axis

def _format_eigenvalues(eigens_list: List, evals_per_gen: int) -> Tuple:
    """
    _format_eigenvalues() takes iter matrices of size gen x dim.
    Each matrix is returned by a single algorithm run. Each row corresponds to an algorithm generation, each row to an eigenvalue.
    The average of matrices is computed, then each column is exported as a list to be plotted. One x-axis is created for all plots.
    The return value is a tuple of two lists. The first one represents the x-axis.
    The second is a list of other axes, each being a separate list, like the x-axis.
    """
    (gen1, dim1) = eigens_list[0].shape
    iterations = len(eigens_list)
    sum = np.zeros((gen1, dim1))
    for eigens_matrix in eigens_list:
        assert  eigens_matrix.shape == (gen1, dim1), "Runs are of different length, cannot take average of sigma"
        sum += eigens_matrix
    sum /= iterations
    x_axis = [x*evals_per_gen for x in range(gen1)]
    other_axes = []
    for i in range(dim1):
        other_axes.append(sum[:,i])
    return x_axis, other_axes

def _format_condition_numbers(conds_list: List, evals_per_gen: int) -> Tuple:
    """
    _format_diffs() takes difference data collected from multiple algorithm runs.
    Then, an average is computed and horizontal axis scaling is applied.
    The return value is a tuple of two list, each corresponds to a plot axis, ready to be plotted.
    """
    max_length = len(conds_list[0])
    for i in conds_list:
        assert len(i) == max_length, "Runs are of different length, cannot take average of condition numbers"
    y_axis = []
    for i in range(max_length):
        y_axis.append(sum([conds[i] for conds in conds_list if conds[i] is not None]) / len(conds_list))
    x_axis = [x*evals_per_gen for x in range(max_length)]
    return x_axis, y_axis

def run_test(dimensions: int = 10, iterations: int = 10, lbd: int = None, visual: bool = False):
    runsc = [
        (None,         False),
        ('reflection', False),
        ('projection', True),
        ('resampling', False)
    ]
    ecdf_plots = []
    sigma_plots = []
    diff_plots = []
    eigen_plots = []
    cond_plots = []

    for rmode, v in runsc:
        ecdf, sigma, diff, eigen, cond = evaluate(rmode, dimensions, iterations, None, lbd, visual and v)
        ecdf_plots.append((ecdf[0], ecdf[1], str(rmode)))
        sigma_plots.append((sigma[0], sigma[1], str(rmode)))
        diff_plots.append((diff[0], diff[1], str(rmode)))
        eigen_plots.append((eigen[0], eigen[1], str(rmode)))
        cond_plots.append((cond[0], cond[1], str(rmode)))

    lambda_prompt = str(lbd) if lbd is not None else "default"
    title = f"dimensions: {dimensions}; iterations: {iterations}; population: {lambda_prompt}"
    ecdf_ax = plt.subplot(411)
    plt.title(title, fontsize=18)
    plt.setp(ecdf_ax.get_xticklabels(), visible = False)
    for ecdf_plot in ecdf_plots:
        plt.plot(ecdf_plot[0], ecdf_plot[1], label=ecdf_plot[2])
    plt.legend(fontsize=12)
    plt.ylabel("ECDF values")
    plt.ylim(0,1)
    
    sigma_ax = plt.subplot(412, sharex=ecdf_ax)
    plt.setp(sigma_ax.get_xticklabels(), visible = False)
    for sigma in sigma_plots:
        plt.plot(sigma[0], sigma[1], label=sigma[2])
    plt.yscale("log")
    plt.ylabel("sigma values")

    diff_ax = plt.subplot(413, sharex=ecdf_ax)
    plt.setp(diff_ax.get_xticklabels(), fontsize = 12)
    for diff in diff_plots:
        plt.plot(diff[0], diff[1], label=diff[2])
    plt.ylabel("sigma difference values")
    plt.xlabel("# of function evaluations", fontsize=12)

    cond_ax = plt.subplot(414, sharex=ecdf_ax)
    plt.setp(cond_ax.get_xticklabels(), fontsize = 12)
    for cond in cond_plots:
        plt.plot(cond[0], cond[1], label=cond[2])
    plt.ylabel("condition number values")
    plt.xlabel("# of function evaluations", fontsize=12)

    fig, axs = plt.subplots(2,2, sharex = True, sharey=True)
    plt.yscale("log")
    fig.subplots_adjust(hspace=0, wspace=0)

    eigen_plot = eigen_plots[0]
    for axis in eigen_plot[1]:
        axs[0][0].plot(eigen_plot[0], axis)
    axs[0][0].title.set_text(eigen_plot[2])

    eigen_plot = eigen_plots[1]
    for axis in eigen_plot[1]:
        axs[0][1].plot(eigen_plot[0], axis)
    axs[0][1].title.set_text(eigen_plot[2])

    eigen_plot = eigen_plots[2]
    for axis in eigen_plot[1]:
        axs[1][0].plot(eigen_plot[0], axis)
    axs[1][0].title.set_text(eigen_plot[2])

    eigen_plot = eigen_plots[3]
    for axis in eigen_plot[1]:
        axs[1][1].plot(eigen_plot[0], axis)
    axs[1][1].title.set_text(eigen_plot[2])

    plt.show()


if __name__ == "__main__":
    run_test()