import matplotlib.pyplot as plt

from pymoo.algorithms.moo.omni import OmniOptimizer
from pymoo.optimize import minimize
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.visualization.scatter import Scatter

# The omni-test problem has 3^n_var equivalent Pareto subsets in the decision space that
# all map to the same Pareto front. The key feature of the Omni-Optimizer is the crowding
# distance defined in the variable space, which allows it to find and maintain all of those
# subsets. Disabling it (var_crowding=False) niches only in objective space and essentially
# recovers the NSGA-II behavior - the equivalent subsets then collapse onto a few of them.
problem = OmniTest(n_var=2)

with_var = minimize(problem, OmniOptimizer(pop_size=100),
                    ('n_gen', 200), seed=1, verbose=False)

without_var = minimize(problem, OmniOptimizer(pop_size=100, var_crowding=False),
                       ('n_gen', 200), seed=1, verbose=False)

PS = problem.pareto_set(5000)
PF = problem.pareto_front(5000)

# decision space - with variable-space niching all equivalent subsets are covered
plot = Scatter(title="Decision Space (with variable-space niching)", labels="x")
plot.add(PS, s=10, color="red", label="Pareto set")
plot.add(with_var.X, s=20, color="blue", label="Obtained solutions")
plot.do()
plt.legend()

# decision space - without it, only a subset of the equivalent regions survives
plot = Scatter(title="Decision Space (without variable-space niching)", labels="x")
plot.add(PS, s=10, color="red", label="Pareto set")
plot.add(without_var.X, s=20, color="blue", label="Obtained solutions")
plot.do()
plt.legend()

# both converge to the same Pareto front in objective space
plot = Scatter(title="Objective Space")
plot.add(PF, s=10, color="red", label="Pareto front")
plot.add(with_var.F, s=20, color="blue", label="Obtained solutions")
plot.do()
plt.legend()

plt.show()
