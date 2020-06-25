from pymoo.algorithms.nsga3 import NSGA3
from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize
from pymoo.problems.multi.sympart import SYMPART, SYMPARTRotated
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt


for problem, name in zip([SYMPART(), SYMPARTRotated()], ["SYM-PART", "SYM-PART rotated"]):

    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=20)
    PS = problem.pareto_set(500)
    PF = problem.pareto_front(500)

    algorithm = NSGA3(ref_dirs=ref_dirs)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 500),
                   seed=1,
                   verbose=False)

    fig_name = f"{algorithm.__class__.__name__} on {name}"
    # visualize decision space
    plot = Scatter(title="Decision Space")
    plot.add(PS, s=10, color='r', label="PS")
    plot.add(res.X, s=30, color='b', label="Obtained solutions")
    plot.do()
    plt.legend()

    # visualize objective space
    plot = Scatter(title="Objective Space")
    plot.add(PF, s=10, color='r', label="PF")
    plot.add(res.F, s=30, color='b', label="Obtained solutions")
    plot.do()
    plt.legend()

    plt.show()
