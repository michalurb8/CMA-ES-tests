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