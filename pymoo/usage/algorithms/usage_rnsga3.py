# START rnsga3
import numpy as np

from pymoo.algorithms.rnsga3 import rnsga3
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util import plotting

# Get problem
from pymoo.util.reference_direction import UniformReferenceDirectionFactory

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
# END rnsga3

# START rnsga3_3d
import matplotlib.pyplot as plt

# Get problem
problem = get_problem("dtlz4", n_var=12, n_obj=3)

# Define reference points and reference directions
ref_points = np.array([[1.0, 0.5, 0.2], [0.3, 0.2, 0.6]])
ref_dirs = UniformReferenceDirectionFactory(3, n_points=91).do()
pf = problem.pareto_front(ref_dirs)

# Get Algorithm
method = rnsga3(
    ref_points=ref_points,
    pop_per_ref_point=91,
    mu=0.1)

res = minimize(problem,
               method=method,
               termination=('n_gen', 300),
               pf=pf,
               disp=False)

ax = plotting.plot(res.F, pf, ref_points, show=False,
              labels=['F', 'pf', 'ref_points'])
ax.view_init(45, 45)
plt.legend()
plt.show()
# END rnsga3_3d