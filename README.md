# AMHE-CMA-ES
An implementation of CMA-ES algorithm for testing bound constraint repair methods.

Required Python3 libraries:
numpy 1.20.2
matplotlib 3.4.0

Run with:
python3 main.py
This will run the default test on all variants of the algorithm and plot resulting ECDF curves, sigma values, covariance matrix eigenvalues, objective function of population mean, covariance matrix condition number.

All testing parameters can be set manually. For help, run:
python3 main.py -h

All possible arguments:
-h Show information about all possible arguments.
-d An integer, number of dimensions. Default is 10.
-i An integer, number of iterations to take average of. Default is 10.
-l An integer, number of points in the population. Default is 4*number of dimensions.
-s An integer, number of generations of a single CMA-ES run. Default is 50.
-v Include to turn on visual mode, plotting all points of each CMA-ES generation.
-c Inlucde to turn on delta correction.

Example uses:

python3 main.py -d 2 -v
Runs the algorithm in 2 dimensions in visual mode.

python3 main.py -i 100
Runs the algorithm with the default dimensionality, population, stop condition, without visuals, 100 times.