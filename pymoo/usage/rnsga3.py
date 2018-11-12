import numpy as np

from pymoo.optimize import minimize
from pymoo.util import plotting
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.factory import get_problem

problem = get_problem("zdt1", n_var=30)

# create the reference directions to be used for the optimization
ref_dirs = UniformReferenceDirectionFactory(2, n_points=92).do()

# create the pareto front for the given reference lines
pf = problem.pareto_front(100)

res = minimize(problem,
               method='rnsga3',
               method_args={
                   'ref_points': np.array([[0.2, 0.7], [0.7, 0.2]]),
                   'pop_per_ref_point': 20},
               termination=('n_gen', 400),
               pf=pf,
               disp=True)
plotting.plot(res.F)

