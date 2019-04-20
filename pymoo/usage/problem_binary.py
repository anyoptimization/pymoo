import numpy as np

from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymop import create_random_knapsack_problem

problem = create_random_knapsack_problem(30)
problem.type_var = np.bool

method = get_algorithm("ga",
                       pop_size=200,
                       sampling=get_sampling("bin_random"),
                       crossover=get_crossover("bin_hux"),
                       mutation=get_mutation("bin_bitflip"),
                       elimate_duplicates=True)


res = minimize(problem,
               method,
               termination=('n_gen', 100),
               disp=True)

print("Best solution found: %s" % res.X.astype(np.int))
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)
