# AMHE-CMA-ES
An implementation of CMA-ES algorithm for testing bound constraint repair methods.

Required Python3 libraries:
numpy 1.20.2
matplotlib 3.4.0

Run with:
python3 main.py
This will run the default test on all variants of the algorithm and show resulting ECDF curves, sigma values and covariance matrix eigenvalues.

All testing parameters can be set manually. For help, run:
python3 main.py -h

All possible arguments:
-h Show information about possible arguments and their values.
-d An integer, number of dimensions. Default is 10.
-i An integer, number of iterations to calculate mean. Default is 10.
-l An integer, number of points in the population. Default is 4*number of dimensions.
-v Include to turn on visual mode, plotting all points of each CMA-ES iteration.

Example uses:

python3 main.py -d 2 -v
Runs the algorithm in 2 dimensions with visual mode.

python3 main.py -i 100
Runs the algorithm with the default 10 dimensions, population of 40, for 100 iterations.