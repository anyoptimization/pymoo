# ZDT 1
#
import numpy as np

from pymoo.optimize import minimize
from pymoo.util import plotting
from pymop.factory import get_problem

problem = get_problem("zdt1", n_var=30)
pf = problem.pareto_front()

# create the reference directions to be used for the optimization
ref_points = np.array([[0.5, 0.2], [0.1, 0.6]])

res = minimize(problem,
               method='rnsga2',
               method_args={
                   'pop_size': 40,
                   'ref_points': ref_points,
                   'epsilon': 0.02,
                   'normalization': 'no',
                   'survival_type': "closest",
                   'extreme_points_as_reference_points': False
                   # 'weights': np.array([0.9, 0.1])
               },
               save_history=True,
               termination=('n_gen', 500),
               seed=1,
               pf=pf,
               disp=True)

print(res.pop.get("dist_to_closest"))
plotting.plot(pf, res.pop.get("F"), ref_points, show=True, labels=['pf', 'F', 'ref_points'])
