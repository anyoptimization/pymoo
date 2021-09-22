import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.constraints.adaptive import AdaptiveConstraintHandling
from pymoo.core.problem import Problem
from pymoo.factory import get_problem
from pymoo.problems.single import G1


class ConstrainedProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=5, n_obj=1, n_ieq_constr=2, n_eq_constr=2, xl=0, xu=1, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (x ** 2).sum(axis=1)

        g1 = 0.25 - x[:, 2] ** 2
        g2 = 0.25 - x[:, 3] ** 2
        out["G"] = np.column_stack([g1, g2])

        h1 = x[:, 1] - x[:, 0]
        h2 = x[:, 1] + x[:, 0] - 1
        out["H"] = np.column_stack([h1, h2])


from pymoo.default import Default

problem = ConstrainedProblem()

problem = get_problem("g05")

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize

algorithm = DE(
    pop_size=100,
    sampling=LHS(),
    variant="DE/rand/1/bin",
    CR=0.3,
    dither="vector",
    jitter=False
)


# algorithm = AdaptiveConstraintHandling(algorithm)

res = minimize(problem,
               algorithm,
               ("n_gen", 1000),
               seed=1,
               verbose=True)


print("Best solution found: \nX = %s\nF = %s\nCV=%s" % (res.X, res.F, res.CV))
print(problem.pareto_front())

exit()
# import numpy as np
# from pymoo.factory import get_problem, get_visualization
#
# problem = get_problem("sphere", n_var=1)
# get_visualization("fitness-landscape", problem, n_samples=1000, title="Sphere").show()


from pymoo.optimize import minimize


class ConstrainedProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=5, n_obj=1, n_ieq_constr=2, n_eq_constr=2, xl=0, xu=1, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (x ** 2).sum(axis=1)

        g1 = 0.25 - x[:, 2] ** 2
        g2 = 0.25 - x[:, 3] ** 2
        out["G"] = np.column_stack([g1, g2])

        h1 = x[:, 1] - x[:, 0]
        h2 = x[:, 1] + x[:, 0] - 1
        out["H"] = np.column_stack([h1, h2])


problem = ConstrainedProblem()

# problem = G5()

# problem = BNH()

algorithm = ISRES(n_offsprings=200, rule=1.0 / 7.0, gamma=0.85, alpha=0.2)

algorithm = GA()

algorithm = DE(variant="DE/rand/1/bin")

Default.tcv.ieq_scale = np.array([0.1, 0.2])

Default.tcv.eq_scale = np.array([0.1, 0.2])

import time

start_time = time.clock()

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)

print(res.F, res.X)

print(time.clock() - start_time, "seconds")

# algorithm = DE()
#
# for k in range(1, 25):
#     label = "g%02d" % k
#
#     problem = get_problem(label)
#
#     res = minimize(problem,
#                    algorithm,
#                    ('n_gen', 1000),
#                    seed=1,
#                    verbose=False)
#
#     print(label, problem.pareto_front(), res.F, res.X)
