# AMHE-CMA-ES
A modification of CMA-ES for AMHE classes

Required Python3 libraries:
numpy 1.20.2
matplotlib 3.4.0

Run with:
python3 main.py
This will test all variants of the algorithm and show resulting ECDF curves.
May take a long time.

All testing parameters can be set manually. For help, run:
python3 main.py -h

All possible arguments:
-h Show information about possible arguments and their values.
-t What test should be run. Possible values: 'all', 'mean_all', 'mean_selected'. Default is 'all'.
-d An integer, number of dimensions. Default is 10.
-i An integer, number of iterations to calculate mean ECDF from. Default is 100.
-l An integer, number of points in the population. Default is 100.
-f List of strings, which objective functions should to be averaged for the result. Possible values: 'felli', 'quadratic', 'bent', 'rastrigin', 'rosenbrock'.
-m Whether to run default CMA-ES algorithm or modified version. Possible values: 'normal', 'mean_all', 'mean_selected'.
-fr How many iteration apart should modification of the algorithm take place. Ignored if 'normal' mode is selected.

There are 5 test cases that can be run:
- selected_frequency - testing mean_selected modification with frequencies 1, 5, 50
- all_frequency - testing mean_all modification with frequencies 1, 5, 50
- modifications_test - testing mean_all_1, mean_selected_1, normal
- all_test - testing all 7 default scenarios:
    normal, mean_all_1, mean_all_5, mean_all_50, mean_selected_1, mean_selected_5, mean_selected_50
- custom - testing one fully customizable run

Use examples:

python3 main.py -t 'all' -i 50 -d 8
Shows ECDF curves of 7 default test cases in 8 dimensions, averaged from 50 iterations.

python3 main.py -t 'custom' -m 'normal' -l 50 -f 'rastrigin'
Shows one ECDF curve of normal (not modified) CMA-ES with population size 50, objective function being rastrigin, all other parameters default.