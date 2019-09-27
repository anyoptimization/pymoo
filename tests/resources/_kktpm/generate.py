from pymoo.optimize import minimize
from pymop.factory import get_problem
import numpy as np


def generate_test_data():
    for str_problem in ["osy"]:
        problem = get_problem(str_problem)

        X = []

        # define a callback function that prints the X and F value of the best individual
        def my_callback(algorithm):
            pop = algorithm.pop
            _X = pop.get("X")[np.random.permutation(len(pop))[:10]]
            X.append(_X)

        minimize(problem,
                 method='nsga2',
                 method_args={'pop_size': 100},
                 termination=('n_gen', 100),
                 callback=my_callback,
                 pf=problem.pareto_front(),
                 disp=True,
                 seed=1)

        np.savetxt("%s.x" % str_problem, np.concatenate(X, axis=0), delimiter=",")

generate_test_data()
