import numpy as np
from optproblems.wfg import WFG3 as WFG3opt

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.problems.multi import WFG3
from pymoo.visualization.scatter import Scatter

n_obj = 3
n_var = 6

np.random.seed(1)
phenome = np.random.random(n_var)

phenome = [0.4851565977955035, 0.053878298054392586, 2.298833100264349, 3.3172215238759395, 3.5, 4.199999999999999]

f = WFG3opt(n_obj, n_var, 4).objective_function(phenome)
_f = WFG3(n_var, n_obj, 4).evaluate(phenome)


for m in range(2, 3, 3):
    for n in range(6, 24, 6):
        for k in range(1, 10):
            name = "wfg%s" % k

            problems = [get_problem(name, n, m) ]

            for problem in problems:

                algorithm = NSGA2(pop_size=1000)

                res = minimize(problem,
                               algorithm,
                               seed=1,
                               termination=('n_gen', 500))

                Scatter(title=name).add(res.F, s=10).show()

                print()
