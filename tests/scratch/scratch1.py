from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
import numpy as np

problem = get_problem("zdt3", n_var=5)

np.random.seed(10)
print(np.random.random())


def my_callback(method):
    X = np.loadtxt("gen_%s.csv" % method.n_gen)
    _X = method.pop.get("X")
    if np.any(np.abs(X - _X) > 1e-8):
        print("test")


algorithm = NSGA2(pop_size=100,
                  callback=my_callback,
                  eliminate_duplicates=False)

res = minimize(problem,
               algorithm,
               ('n_gen', 20),
               seed=1,
               verbose=True)
