from cmaes import CMAES
import numpy as np
import matplotlib.pyplot as plt

_TARGETS = np.array([10**i for i in range(-10, 10)])

def evaluate(dimensions: int, mode: str, frequency: int, iterations: int, lambda_arg: int):
        ecdf_list = []
        print("Starting evaluation...")
        print("mode:", mode, "dimensions:", dimensions, ", frequency:", frequency, ", iterations:", iterations, ", population:", lambda_arg)
        for objective in ['quadratic', 'felli', 'bent']:
            print("    Currently running:", objective)
            for i in range(iterations):
                algo = CMAES(objective, dimensions, mode, frequency, lambda_arg)
                algo.generation_loop()
                ecdf_list.append(algo.ecdf(_TARGETS))
        max_length = max([len(ecdf) for ecdf in ecdf_list])

        for ecdf in ecdf_list: #fill ecdf data with 1s so that all lists are of equal lengths
            missing_entries = max_length - len(ecdf)
            if missing_entries > 0:
                ecdf.extend([1.]*missing_entries)
        
        ecdf_result = []
        for i in range(max_length):
            ecdf_result.append(sum([ecdf[i] for ecdf in ecdf_list])/len(ecdf_list))
        return ecdf_result

def selected_frequency_test(dimensions: int, iterations: int, lambda_arg: int):
    ecdf_ms_1 = evaluate(dimensions, 'mean_selected', 1, iterations, lambda_arg)
    ecdf_ms_5 = evaluate(dimensions, 'mean_selected', 5, iterations, lambda_arg)
    ecdf_ms_50 = evaluate(dimensions, 'mean_selected', 50, iterations, lambda_arg)
    plt.plot(ecdf_ms_1, label = 'mean_selected 1')
    plt.plot(ecdf_ms_5, label = 'mean_selected 5')
    plt.plot(ecdf_ms_50, label = 'mean_selected 50')
    plt.legend()
    plt.show()

def all_frequency_test(dimensions: int, iterations: int, lambda_arg: int):
    ecdf_ma_1 = evaluate(dimensions, 'mean_all', 1, iterations, lambda_arg)
    ecdf_ma_5 = evaluate(dimensions, 'mean_all', 5, iterations, lambda_arg)
    ecdf_ma_50 = evaluate(dimensions, 'mean_all', 50, iterations, lambda_arg)
    plt.plot(ecdf_ma_1, label = 'mean_all 1')
    plt.plot(ecdf_ma_5, label = 'mean_all 5')
    plt.plot(ecdf_ma_50, label = 'mean_all 50')
    plt.legend()
    plt.show()

def modifications_test(dimensions: int, iterations: int, lambda_arg: int):
    ecdf_normal = evaluate(dimensions, 'normal', None, iterations, lambda_arg)
    ecdf_mean_all = evaluate(dimensions, 'mean_all', 1, iterations, lambda_arg)
    ecdf_mean_selected = evaluate(dimensions, 'mean_selected', 1, iterations, lambda_arg)
    plt.plot(ecdf_normal, label = 'normal')
    plt.plot(ecdf_mean_all, label = 'mean_all 1')
    plt.plot(ecdf_mean_selected, label = 'mean_selected 1')
    plt.legend()
    plt.show()

def all_test(dimensions: int, iterations: int, lambda_arg: int):
    ecdf_ms_1 = evaluate(dimensions, 'mean_selected', 1, iterations, lambda_arg)
    ecdf_ms_5 = evaluate(dimensions, 'mean_selected', 5, iterations, lambda_arg)
    ecdf_ms_50 = evaluate(dimensions, 'mean_selected', 50, iterations, lambda_arg)
    ecdf_normal = evaluate(dimensions, 'normal', None, iterations, lambda_arg)
    ecdf_ma_1 = evaluate(dimensions, 'mean_all', 1, iterations, lambda_arg)
    ecdf_ma_5 = evaluate(dimensions, 'mean_all', 5, iterations, lambda_arg)
    ecdf_ma_50 = evaluate(dimensions, 'mean_all', 50, iterations, lambda_arg)
    plt.plot(ecdf_ms_1, label = 'mean_selected 1')
    plt.plot(ecdf_ms_5, label = 'mean_selected 5')
    plt.plot(ecdf_ms_50, label = 'mean_selected 50')
    plt.plot(ecdf_normal, label = 'normal')
    plt.plot(ecdf_ma_1, label = 'mean_all 1')
    plt.plot(ecdf_ma_5, label = 'mean_all 5')
    plt.plot(ecdf_ma_50, label = 'mean_all 50')
    plt.legend()
    plt.show()