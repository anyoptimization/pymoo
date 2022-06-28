
import numpy as np

from pymoo.problems.functional import FunctionalProblem

# the number of variables the problem has
n_var = 10

# each objective as a function
objs = [
    lambda x: np.sum((x - 2) ** 2),
    lambda x: np.sum((x + 2) ** 2)
]

# each inequality constraint as a function
constr_ieq = [
    lambda x: np.sum((x - 1) ** 2)
]


# now put everything together to a problem object
problem = FunctionalProblem(n_var,
                            objs,
                            constr_ieq=constr_ieq,
                            xl=np.array([-10, -5, -10]),
                            xu=np.array([10, 5, 10])
                            )

F, G = problem.evaluate(np.random.rand(3, 10))
print(f"F: {F}\n")
print(f"G: {G}\n")
