# AMHE-CMA-ES
A modification of CMA-ES for AMHE classes

Required Python3 libraries:
numpy 1.20.2
matplotlib 3.4.0

Run with:
python3 main.py
This will test all variants of the algorithm and show resulting ECDF curves.
May take a long time.

Possible arguments:
-h Show information about possible arguments and their values.
-t What test should be run. Possible values: 'all', 'mean_all', 'mean_selected'. Default is 'all'.
-d An integer, number of dimensions. Default is 10.
-i An integer, number of iterations to calculate mean ECDF from. Default is 100.
-l An integer, number of points in the population. Default is 100.
-f List of strings, which objective functions should to be averaged for the result. Possible values: 'felli', 'quadratic', 'bent', 'rastrigin', 'rosenbrock'.
-m Whether to run default CMA-ES algorithm or modified version. Possible values: 'normal', 'mean_all', 'mean_selected'.
-fr How many iteration apart should modification of the algorithm take place. Ignored if 'normal' mode is selected.