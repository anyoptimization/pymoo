from pymoo.optimize import minimize
from pymoo.util import plotting
from pymop.factory import get_problem, UniformReferenceDirectionFactory

# create the optimization problem
problem = get_problem("dtlz2")
ref_dirs = UniformReferenceDirectionFactory(3, n_points=100).do()
pf = problem.pareto_front(ref_dirs)

res = minimize(problem,
               method='moead',
               method_args={
                   'ref_dirs': ref_dirs,
                   'n_neighbors': 15,
                   'decomposition': 'pbi',
                   'prob_neighbor_mating': 0.7
               },
               termination=('n_gen', 200),
               pf=pf,
               save_history=False,
               disp=True)
plotting.plot(pf, res.F, labels=["Pareto-front", "F"])
