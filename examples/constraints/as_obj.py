from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.constraints.as_obj import ConstraintsAsObjective
from pymoo.optimize import minimize
from pymoo.problems.single import G1
from pymoo.visualization.scatter import Scatter

problem = G1()

problem = ConstraintsAsObjective(problem)

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 300),
               seed=1,
               verbose=True)

plot = Scatter()
plot.add(problem.pareto_front(), marker="*", color="black", alpha=0.7, s=100)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()
