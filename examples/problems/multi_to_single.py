import numpy as np

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT1
from pymoo.problems.multi_to_single import MultiToSingleObjective
from pymoo.visualization.scatter import Scatter

decomposition = Tchebicheff()
kwargs = dict(weights=np.array([0.7, 0.3]))

problem = ZDT1(n_var=30)

problem = MultiToSingleObjective(problem, decomposition, kwargs=kwargs)

algorithm = DE(pop_size=100)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

F = res.opt.get("__F__")

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(F, color="red", marker="x", s=100)
plot.show()
