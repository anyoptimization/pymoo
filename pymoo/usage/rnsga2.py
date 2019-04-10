# ZDT 1
#
import numpy as np

from pymoo.optimize import minimize
from pymoo.util import plotting
from pymop.factory import get_problem

problem = get_problem("zdt1", n_var=30)
pf = problem.pareto_front()

# create the reference directions to be used for the optimization
ref_points = np.array([[0.1, 0.6], [0.5, 0.2]])

res = minimize(problem,
               method='rnsga2',
               method_args={
                   'pop_size': 100,
                   'ref_points': ref_points,
                   'epsilon': 0.01,
                   'normalization': 'ever'
               },
               save_history=True,
               termination=('n_gen', 200),
               seed=1,
               pf=pf,
               disp=True)

plotting.plot(pf, res.pop.get("F"), ref_points, show=True, labels=['pf', 'F', 'ref_points'])
