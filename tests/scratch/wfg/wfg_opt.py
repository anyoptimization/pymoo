from optproblems.wfg import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
import numpy as np

from pymoo.factory import get_problem
from pymoo.performance_indicator.igd import IGD
from pymoo.visualization.scatter import Scatter

n_obj = 3

problems = [
    (get_problem("wfg1", 6, n_obj, 4), WFG1(n_obj, 6, 4)),
    (get_problem("wfg2", 6, n_obj, 4), WFG2(n_obj, 6, 4)),
    (get_problem("wfg3", 6, n_obj, 4), WFG3(n_obj, 6, 4)),
    (get_problem("wfg4", 6, n_obj, 4), WFG4(n_obj, 6, 4)),
    (get_problem("wfg5", 6, n_obj, 4), WFG5(n_obj, 6, 4)),
    (get_problem("wfg6", 6, n_obj, 4), WFG6(n_obj, 6, 4)),
    (get_problem("wfg7", 6, n_obj, 4), WFG7(n_obj, 6, 4)),
    (get_problem("wfg8", 6, n_obj, 4), WFG8(n_obj, 6, 4)),
    (get_problem("wfg9", 6, n_obj, 4), WFG9(n_obj, 6, 4)),
]

for my, other in problems:

    ps = other.get_optimal_solutions(2000)
    for e in ps:
        e.objective_values = other.objective_function(e.phenome)
    pf = np.array([e.objective_values for e in ps])
    ps = np.array([e.phenome for e in ps])

    _ps = my.pareto_set(n_pareto_points=3000)
    _pf = my.pareto_front(n_pareto_points=3000)

    name = my.__class__.__name__
    Scatter(title=name).add(pf, s=15, color="green", alpha=0.5).add(_pf, color="red", s=10).show()

    print(name, IGD(pf).calc(_pf))

    print()
