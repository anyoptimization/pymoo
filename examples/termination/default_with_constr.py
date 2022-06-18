from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination

problem = get_problem("g5")
algorithm = GA(pop_size=100)
termination = DefaultSingleObjectiveTermination()

res = minimize(problem,
               algorithm,
               termination,
               return_least_infeasible=True,
               pf=None,
               seed=1,
               verbose=True)

print("n_gen: ", res.algorithm.n_gen)
print("CV: ", res.CV[0])
print("F: ", res.F[0])

