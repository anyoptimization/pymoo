from pymoo.algorithms.soo.nonconvex.ga_niching import NicheGA
from pymoo.optimize import minimize
from pymoo.problems.single import Rastrigin
from pymoo.problems.single.simple import SimpleMultiModal01

problem = SimpleMultiModal01()

problem = Rastrigin(n_var=30)

algorithm = NicheGA(pop_size=100)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)


print(res.F)
