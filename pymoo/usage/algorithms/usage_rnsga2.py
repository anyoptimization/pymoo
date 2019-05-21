import numpy as np

from pymoo.algorithms.rnsga2 import rnsga2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util import plotting

# Get problem
problem = get_problem("zdt1", n_var=30)
pf = problem.pareto_front()

# Define reference points
ref_points = np.array([[0.5, 0.2], [0.1, 0.6]])

# Get Algorithm
method = rnsga2(
    ref_points=ref_points,
    pop_size=40,
    epsilon=0.01,
    normalization='front',
    extreme_points_as_reference_points=False,
    weights=np.array([0.5, 0.5]))

res = minimize(problem,
               method=method,
               save_history=True,
               termination=('n_gen', 250),
               seed=1,
               pf=pf,
               disp=False)

plotting.plot(pf, res.F, ref_points, show=True, labels=['pf', 'F', 'ref_points'])
