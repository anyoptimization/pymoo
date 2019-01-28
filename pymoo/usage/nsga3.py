import matplotlib.pyplot as plt

from pymoo.optimize import minimize
from pymoo.util import plotting
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.factory import get_problem

problem = get_problem("c3dtlz4", n_var=12, n_obj=3)

# create the reference directions to be used for the optimization
ref_dirs = UniformReferenceDirectionFactory(3, n_points=91).do()
#ref_dirs = UniformReferenceDirectionFactory(2, n_points=100).do()

# create the pareto front for the given reference lines
pf = problem.pareto_front(ref_dirs)

res = minimize(problem,
               method='nsga3',
               method_args={
                   'pop_size': 92,
                   'ref_dirs': ref_dirs
               },
               termination=('n_gen', 1000),
               pf=pf,
               seed=4,
               disp=True)

plotting.plot(res.F)
