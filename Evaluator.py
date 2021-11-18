from cmaes import CMAES
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sys import stdout

_TARGETS = np.array([10 ** i for i in range(-10, 10)])

def evaluate(repair_mode: str, dimensions: int, iterations: int, objectives: List, lambda_arg: int, stop_after: int, visual: bool, correction: bool):
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
    stop_after: int
        How many iterations are to be run
    visual: bool
        If True, every algorithm generation will be visualised (only 2 first dimensions)
    ----------
    # run the algorithm multiple times, return averaged results: ECDF values, sigma values, sigma difference values
    """
    if objectives is None:
            objectives = ['rastrigin']
    ecdfs_list = []
    sigmas_list = []
    diffs_list = []
    eigens_list = []
    cond_list = []
    mean_list = []
    repair_list = []
    evals_per_gen = None
    print("Starting evaluation...")
    lambda_prompt = str(lambda_arg) if lambda_arg is not None else "default"
    print(f"dimensions: {dimensions}; iterations: {iterations}; population: {lambda_prompt}; repair: {repair_mode}; correction: {correction}")
    for objective in objectives:
        print("    Currently running:", objective)
        for iteration in range(iterations):
            stdout.write(f"\rIteration: {1+iteration} / {iterations}")
            stdout.flush()
            algo = CMAES(objective, dimensions, repair_mode, lambda_arg, stop_after, visual, correction) # algorithm runs here
            algo.generation_loop()
            if evals_per_gen == None:
                evals_per_gen = algo.evals_per_iteration()
            else:
                assert evals_per_gen == algo.evals_per_iteration(), "Lambda different for same settings"
            ecdfs_list.append(algo.ecdf_history(_TARGETS))
            sigmas_list.append(algo.sigma_history())
            diffs_list.append(algo.diff_history())
            eigens_list.append(algo.eigen_history())
            cond_list.append(algo.cond_history())
            mean_list.append(algo.mean_history())
            repair_list.append(algo.repair_history())
        print()

    formatted_ecdfs = _format_list(ecdfs_list, evals_per_gen)
    formatted_sigmas = _format_list(sigmas_list, evals_per_gen)
    formatted_diffs = _format_list(diffs_list, evals_per_gen)
    formatted_conds = _format_list(cond_list, evals_per_gen)
    formatted_means = _format_list(mean_list, evals_per_gen)
    formatted_repairs = _format_list(repair_list, evals_per_gen)
    formatted_eigens = _format_eigenvalues(eigens_list, evals_per_gen)
    return (formatted_ecdfs, formatted_sigmas, formatted_diffs, formatted_eigens, formatted_conds, formatted_means, formatted_repairs)


def _format_list(input_list: List, evals_per_gen: int) -> Tuple:
    """
    _format_list() takes as input data collected from multiple algorithm runs.
    Then, an average is computed and horizontal axis scaling is applied.
    The return value is a tuple of two list, each corresponds to a plot axis, ready to be plotted.
    """
    run_count = len(input_list)
    run_length = len(input_list[0])
    for i in input_list:
        assert len(i) == run_length, "Runs are of different length, cannot take average"
    y_axis = []
    for i in range(run_length):
        y_axis.append(sum([sigmas[i] for sigmas in input_list]) / run_count)
    x_axis = [x*evals_per_gen for x in range(run_length)]
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

def run_test(dimensions: int, iterations: int, lbd: int, stop_after: int, visual: bool, rmode: str):
    """
    run_test()) is the main function. It runs the evaluate function for all the algorithm varianst.
    All of them are plotted separately to be compared.
    """
    ecdf_plots = []
    sigma_plots = []
    diff_plots = []
    eigen_plots = []
    cond_plots = []
    mean_plots = []
    repair_plots = []

    run_params = [(False, False), (True, True)]
    for corr, v in run_params:
        ecdf, sigma, diff, eigen, cond, mean, rep = evaluate(rmode, dimensions, iterations, None, lbd, stop_after, v and visual, corr)

        ecdf_plots.append((ecdf[0], ecdf[1], str(corr)))
        sigma_plots.append((sigma[0], sigma[1], str(corr)))
        diff_plots.append((diff[0], diff[1], str(corr)))
        eigen_plots.append((eigen[0], eigen[1], str(corr)))
        cond_plots.append((cond[0], cond[1], str(corr)))
        mean_plots.append((mean[0], mean[1], str(corr)))
        repair_plots.append((rep[0], rep[1], str(corr)))

    lambda_prompt = str(lbd) if lbd is not None else "default"
    title_str = f"dimensions: {dimensions}; population: {lambda_prompt}; generations: {stop_after}; iterations: {iterations}; method: {rmode}"; 
    ecdf_ax = plt.subplot(611)
    plt.title(title_str, fontsize=14)
    plt.setp(ecdf_ax.get_xticklabels(), visible = False)
    for ecdf_plot in ecdf_plots:
        plt.plot(ecdf_plot[0], ecdf_plot[1], label=ecdf_plot[2])
    plt.legend(fontsize=12)
    plt.ylabel("ECDF")
    plt.ylim(0,1)
    
    sigma_ax = plt.subplot(612, sharex=ecdf_ax)
    plt.setp(sigma_ax.get_xticklabels(), visible = False)
    for sigma in sigma_plots:
        plt.plot(sigma[0], sigma[1], label=sigma[2])
    plt.yscale("log")
    plt.ylabel("sigma")

    diff_ax = plt.subplot(613, sharex=ecdf_ax)
    plt.setp(diff_ax.get_xticklabels(), visible = False)
    for diff in diff_plots:
        plt.plot(diff[0], diff[1], label=diff[2])
    plt.ylabel("sigma difference")

    cond_ax = plt.subplot(614, sharex=ecdf_ax)
    plt.setp(cond_ax.get_xticklabels(), visible = False)
    for cond in cond_plots:
        plt.plot(cond[0], cond[1], label=cond[2])
    plt.ylabel("condition number")

    mean_ax = plt.subplot(615, sharex=ecdf_ax)
    plt.setp(mean_ax.get_xticklabels(), visible = False)
    for mean in mean_plots:
        plt.plot(mean[0], mean[1], label=mean[2])
    plt.ylabel("f(mean)")
    plt.yscale("log")

    rep_ax = plt.subplot(616, sharex=ecdf_ax)
    plt.setp(mean_ax.get_xticklabels(), fontsize = 12)
    for repair in repair_plots:
        plt.plot(repair[0], repair[1], label=repair[2])
    plt.ylabel("# of points repaired")
    plt.xlabel("# of function evaluations", fontsize=12)

    fig, axs = plt.subplots(2,1, sharex = True, sharey=True)
    plt.yscale("log")
    fig.suptitle(title_str, fontsize=14)
    fig.subplots_adjust(hspace=0, wspace=0)

    for index, eigen_plot in enumerate(eigen_plots):
        for eigenvalue in eigen_plot[1]:
            axs[index%2].plot(eigen_plot[0], eigenvalue)
        axs[index%2].title.set_text(eigen_plot[2])

    plt.show()

if __name__ == "__main__":
    run_test()