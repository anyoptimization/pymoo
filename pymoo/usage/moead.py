from pymoo.optimize import minimize
from pymoo.util import plotting
from pymop.factory import get_problem, UniformReferenceDirectionFactory

# create the optimization problem
problem = get_problem("zdt3")
pf = problem.pareto_front()
ref_dirs = UniformReferenceDirectionFactory(2, n_points=100).do()

"""
problem = get_problem("dtlz4", n_var=12, n_obj=3)
# create the reference directions to be used for the optimization
ref_dirs = UniformReferenceDirectionFactory(3, n_points=91).do()
pf = problem.pareto_front(ref_dirs)
"""


# solve the given problem using an optimization algorithm (here: nsga2)
res = minimize(problem,
               method='moead',
               method_args={'ref_dirs': ref_dirs},
               termination=('n_gen', 400),
               pf=pf,
               save_history=False,
               disp=True)
plotting.plot(pf, res.F, labels=["Pareto-front", "F"])