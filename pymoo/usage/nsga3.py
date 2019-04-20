from pymoo.optimize import minimize
from pymoo.util import plotting
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.factory import get_problem
from pymoo.factory import get_algorithm

# create the reference directions to be used for the optimization
ref_dirs = UniformReferenceDirectionFactory(3, n_points=91).do()

# create the algorithm object
method = get_algorithm("nsga3",
                       pop_size=92,
                       ref_dirs=ref_dirs)

# execute the optimization
res = minimize(get_problem("dtlz1"),
               method,
               termination=('n_gen', 200))

plotting.plot(res.F, show=True)
