import numpy as np

from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.core.duplicate import ElementwiseDuplicateElimination


class MyProblem(ElementwiseProblem):

    def __init__(self, my_list):
        self.correct = np.argsort(my_list)
        super().__init__(n_var=len(my_list), n_obj=1, n_ieq_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        pheno = np.argsort(x)
        out["F"] = - float((self.correct == pheno).sum())
        out["pheno"] = pheno
        out["hash"] = hash(str(pheno))


class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return a.get("hash") == b.get("hash")


np.random.seed(2)
list_to_sort = np.random.random(20)
problem = MyProblem(list_to_sort)
print("Sorted by", np.argsort(list_to_sort))

algorithm = BRKGA(
    n_elites=200,
    n_offsprings=700,
    n_mutants=100,
    bias=0.7,
    eliminate_duplicates=MyElementwiseDuplicateElimination())

res = minimize(problem,
               algorithm,
               ("n_gen", 75),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
print("Solution", res.opt.get("pheno")[0])
print("Optimum ", np.argsort(list_to_sort))
