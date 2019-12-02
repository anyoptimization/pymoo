import numpy as np

from pymoo.algorithms.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

n_obj = 2
n_var = 6

np.random.seed(1)
phenome = np.random.random(n_var)

phenome = [0.4851565977955035, 0.053878298054392586, 2.298833100264349, 3.3172215238759395, 3.5, 4.199999999999999]

#f = WFG1opt(n_obj, n_var, 4).objective_function(phenome)
# _f = WFG1(n_var, n_obj, 4).evaluate(phenome)


for m in range(3, 4, 3):
    for n in range(6, 24, 6):
        for k in range(3, 10):
            name = "wfg%s" % k


            problems = [get_problem(name, n, m)]


            for problem in problems:

                ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=24)
                algorithm = NSGA3(ref_dirs)
                pf = problem.pareto_front(ref_dirs)

                res = minimize(problem,
                               algorithm,
                               seed=1,
                               pf=pf,
                               termination=('n_gen', 500),
                               verbose=True)

                Scatter(title=name).add(res.F, s=10).add(res.F, s=5, color="red").show()

                print()
