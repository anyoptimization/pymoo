from pymoo.optimize import minimize
from pymoo.util import plotting
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.factory import get_problem

import numpy as np

problem = get_problem("dtlz1", n_var=7, n_obj=3)
# create the reference directions to be used for the optimization
ref_dirs = UniformReferenceDirectionFactory(3, n_points=105).do()

# create the pareto front for the given reference lines
pf = problem.pareto_front(ref_dirs)

res = minimize(problem,
               method='nsga3',
               method_args={
                   'ref_dirs': ref_dirs
               },
               termination=('n_eval', 45000),
               pf=pf,
               seed=1,
               disp=True)

np.savetxt('test_nsga3_dtlz1.dat', res.F)
plotting.plot(res.F)
