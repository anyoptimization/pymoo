from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.termination import TerminateIfAny
from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT1
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.termination.max_time import TimeBasedTermination
from pymoo.visualization.scatter import Scatter

problem = ZDT1()

algorithm = NSGA2(pop_size=100)

termination = TerminateIfAny(DefaultMultiObjectiveTermination(), TimeBasedTermination(0.2))

res = minimize(problem,
               algorithm,
               termination=termination,
               verbose=True,
               seed=1)

print(res.algorithm.n_gen)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()
