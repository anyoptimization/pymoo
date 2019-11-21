import numpy as np

from pymoo.algorithms.so_brkga import BRKGA
from pymoo.model.duplicate import ElementwiseDuplicateElimination
from pymoo.model.problem import Problem
from pymoo.optimize import minimize


class MyProblem(Problem):

    def __init__(self, my_list):
        self.correct = np.argsort(my_list)
        super().__init__(n_var=len(my_list), n_obj=1, n_constr=0, xl=0, xu=1, elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        pheno = np.argsort(x)
        out["F"] = - float((self.correct == pheno).sum())
        out["pheno"] = pheno


class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return np.all(a.get("pheno") == b.get("pheno"))


np.random.seed(2)
list_to_sort = np.random.random(20)
problem = MyProblem(list_to_sort)

algorithm = BRKGA(eliminate_duplicates=MyElementwiseDuplicateElimination())


res = minimize(problem,
               algorithm,
               ("n_gen", 200),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
print("Solution", res.opt.get("pheno"))
print("Optimum", np.argsort(list_to_sort))

