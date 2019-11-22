from optproblems.wfg import WFG1

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


WFG1
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

                print("test")
