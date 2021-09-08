from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.factory import DTLZ2
from pymoo.optimize import minimize


# problem = get_problem("ackley", n_var=30)
from pymoo.visualization.scatter import Scatter

problem = DTLZ2()

algorithm = AGEMOEA(
    pop_size=100
)

res = minimize(problem,
               algorithm,
               ('n_gen', 300),
               seed=1,
               verbose=True)



Scatter().add(res.F).show()