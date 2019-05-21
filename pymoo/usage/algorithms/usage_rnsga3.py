import numpy as np

from pymoo.algorithms.rnsga3 import rnsga3
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util import plotting

# Get problem
problem = get_problem("zdt1")
pf = problem.pareto_front()

# Define reference points
ref_points = np.array([[0.3, 0.4], [0.8, 0.5]])

# Get Algorithm
method = rnsga3(
    ref_points=ref_points,
    pop_per_ref_point=50,
    mu=0.1)

res = minimize(problem,
               method=method,
               termination=('n_gen', 300),
               pf=pf,
               disp=False)

reference_directions = res.algorithm.survival.ref_dirs
plotting.plot(pf, res.F, ref_points, reference_directions, show=True,
              labels=['pf', 'F', 'ref_points', 'ref_dirs'])
