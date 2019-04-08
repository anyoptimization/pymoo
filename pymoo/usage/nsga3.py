from pymoo.optimize import minimize
from pymoo.util import plotting
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.factory import get_problem

problem = get_problem("dtlz1")

# create the reference directions to be used for the optimization
ref_dirs = UniformReferenceDirectionFactory(3, n_points=91).do()

# create the pareto front for the given reference lines
pf = problem.pareto_front(ref_dirs)

res = minimize(problem,
               method='nsga3',
               method_args={
                   'pop_size': 92,
                   'ref_dirs': ref_dirs
               },
               termination=('n_gen', 600),
               pf=pf,
               seed=4,
               disp=True)

# if desired we can filter out the solutions that are not close to ref_dirs
closest_to_ref_dir = res.opt.get("closest")
plotting.plot(res.F[closest_to_ref_dir, :], show=True)
