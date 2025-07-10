from pymoo.algorithms.moo.cmopso import CMOPSO
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")

algorithm = CMOPSO(
    pop_size=100,
    w=0.729844,
    c1=1.49618,
    c2=1.49618,
    max_velocity_rate=0.2,
    elite_size=10,
    mutate=False,
    seed=1,
)

res = minimize(problem, algorithm, termination=("n_gen", 200), verbose=True)

Scatter().add(res.F).show()
