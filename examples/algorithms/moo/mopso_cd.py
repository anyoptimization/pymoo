from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")

algorithm = MOPSO_CD(
    pop_size=100,
    w=0.729844,
    c1=1.49618,
    c2=1.49618,
    max_velocity_rate=0.2,
    archive_size=200,
    seed=1,
)

res = minimize(problem, algorithm, termination=("n_gen", 200), verbose=True)

Scatter().add(res.F).show()
