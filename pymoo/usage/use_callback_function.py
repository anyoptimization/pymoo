# create the optimization problem
import os
import pickle

from pymoo.optimize import minimize
from pymop.factory import get_problem

problem = get_problem("rastrigin")

# file where a snapshot will be saved
fname = "algorithm.dat"


# define a callback function that prints the X and F value of the best individual
def my_callback(algorithm):

    # get the best individual and print it
    best = algorithm.pop[0]
    print(algorithm.n_gen, best.X, best.F)

    # pickle the algorithm if it might fail for whatever reason
    with open(fname, 'wb') as f:
        pickle.dump(algorithm, f)


res = minimize(problem,
               method='ga',
               method_args={'pop_size': 100},
               termination=('n_gen', 100),
               callback=my_callback)

os.remove(fname)
