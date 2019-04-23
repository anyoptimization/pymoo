import numpy as np

from pymoo.algorithms.rnsga2 import rnsga2
from pymoo.optimize import minimize
from pymoo.util import plotting
from pymop.factory import get_problem

problem = get_problem("zdt1", n_var=30)
pf = problem.pareto_front()

# the reference point to be used during optimization
ref_points = np.array([[0.5, 0.2], [0.1, 0.6]])

method = rnsga2(pop_size=40,
                ref_points=ref_points,
                epsilon=0.05,
                normalization='no',
                extreme_points_as_reference_points=False,
                weights=np.array([0.5, 0.5])
                )

res = minimize(problem,
               method,
               termination=('n_gen', 400),
               disp=True)

plotting.plot(pf, res.F, ref_points, show=True, labels=['pf', 'F', 'ref_points'])
