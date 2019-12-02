import numpy as np

from pymoo.algorithms.nsga2 import RankAndCrowdingSurvival
from pymoo.factory import get_reference_directions
from pymoo.model.population import Population
from pymoo.problems.many import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
from pymoo.visualization.scatter import Scatter


def calc_pareto_front(problem, ref_dirs):
    n_pareto_points = 200
    np.random.seed(1)

    pf = problem.pareto_front(n_pareto_points=n_pareto_points, use_cache=False)
    # survival = ReferenceDirectionSurvival(ref_dirs)
    survival = RankAndCrowdingSurvival()

    for i in range(1000):
        _pf = problem.pareto_front(n_pareto_points=n_pareto_points, use_cache=False)
        F = np.row_stack([pf, _pf])

        pop = Population().new("F", F)
        pop = survival.do(problem, pop, n_pareto_points // 2)

        pf = pop.get("F")

    return pf


if __name__ == '__main__':

    ref_dirs = get_reference_directions("das-dennis", 3, n_points=91)
    F = calc_pareto_front(WFG3(6, 3), ref_dirs)

    Scatter().add(F).show()

    for problem in [WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9]:
        print("")
