from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util.termination.default import SingleObjectiveDefaultTermination

problem = get_problem("g05")
algorithm = GA(pop_size=100)
termination = SingleObjectiveDefaultTermination()

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

