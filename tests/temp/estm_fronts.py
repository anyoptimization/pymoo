import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.population import Population
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

for problem in [get_problem(f"ctp{k}") for k in range(1, 9)]:

    label = problem.__class__.__name__.lower()

    if label != "ctp4":
        continue

    fname = f"{label}.pf"
    #
    # archive = Population()
    #
    # for i in range(10):
    #     algorithm = NSGA2(pop_size=200)
    #
    #     res = minimize(problem,
    #                    algorithm,
    #                    ('n_gen', 3000),
    #                    seed=1,
    #                    verbose=False)
    #
    #     archive = Population.merge(archive, res.opt)
    #
    # opt = RankAndCrowdingSurvival().do(problem, archive, n_survive=1000)
    #
    # pf = opt.get("F")
    #
    # np.savetxt(fname, pf)
    # print(label)

    pf = np.loadtxt(f"../../data/pf/CTP/{fname}")

    plot = Scatter(title=label)
    plot.add(pf, color="red")
    plot.show()

