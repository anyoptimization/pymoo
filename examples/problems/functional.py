
import numpy as np

from pymoo.problems.functional import FunctionalProblem

objs = [
    lambda x: np.sum((x - 2) ** 2),
    lambda x: np.sum((x + 2) ** 2)
]

constr_ieq = [
    lambda x: np.sum((x - 1) ** 2)
]

n_var = 10

problem = FunctionalProblem(n_var,
                            objs,
                            constr_ieq=constr_ieq,
                            xl=np.array([-10, -5, -10]),
                            xu=np.array([10, 5, 10])
                            )

F, G, H = problem.evaluate(np.random.rand(3, 10))

print(f"F: {F}\n")
print(f"G: {G}\n")
print(f"H: {H}")

