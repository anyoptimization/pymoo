from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.util import plotting
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.factory import get_problem

problem = get_problem("dtlz1", n_var=7, n_obj=3)
# create the reference directions to be used for the optimization
ref_dirs = UniformReferenceDirectionFactory(3, n_points=91).do()

# create the pareto front for the given reference lines
pf = problem.pareto_front(ref_dirs)

res = minimize(problem,
               method='nsga3',
               method_args={
                   'pop_size': 92,
                   'ref_dirs': ref_dirs},
               termination=('n_gen', 400),
               pf=pf,
               seed=1,
               disp=True)
plotting.plot(res.F)
