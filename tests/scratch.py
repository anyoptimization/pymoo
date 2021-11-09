from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.factory import get_problem
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize

problem = get_problem("ackley", n_var=10)

algorithm = DE(
pop_size=100,
sampling=LHS(),
variant='DE/current-to-pbest/1/bin')

res = minimize(problem,
algorithm,
seed=1,
verbose=False)




