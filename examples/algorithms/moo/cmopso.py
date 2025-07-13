from pymoo.algorithms.moo.cmopso import CMOPSO
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")

algorithm = CMOPSO(
    pop_size=100,
    max_velocity_rate=0.2,
    elite_size=10,
    mutation_rate=0.5,
    seed=1,
)

res = minimize(problem, algorithm, termination=("n_gen", 200), verbose=True)

Scatter().add(res.F).show()
