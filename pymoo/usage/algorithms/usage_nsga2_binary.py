from pymoo.factory import get_problem, get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import nsga2
from pymoo.util import plotting

method = nsga2(pop_size=100,
               sampling=get_sampling("bin_random"),
               crossover=get_crossover("bin_two_point"),
               mutation=get_mutation("bin_bitflip"),
               elimate_duplicates=True)

res = minimize(get_problem("zdt5"),
               method,
               ('n_gen', 300),
               seed=1,
               verbose=False)

plotting.plot(res.F, no_fill=True)
