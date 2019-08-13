
import numpy as np

# this will be the evaluation function that is called each time
from pymoo.model.problem import get_problem_from_func


def my_evaluate_func(x, out, *args, **kwargs):

    # define the objective as x^2
    f1 = np.sum(np.square(x - 2), axis=1)
    f2 = np.sum(np.square(x + 2), axis=1)
    out["F"] = np.column_stack([f1, f2])

    # x^2 < 2 constraint
    out["G"] = np.sum(np.square(x - 1), axis=1)


# load the problem from a function - define 3 variables with the same lower bound
problem = get_problem_from_func(my_evaluate_func, xl=-10, xu=10, n_var=3)
F, CV = problem.evaluate(np.random.rand(100, 3))

# or define a problem with varying lower and upper bounds
problem = get_problem_from_func(my_evaluate_func, xl=np.array([-10, -5, -10]), xu=np.array([10, 5, 10]))
F, CV = problem.evaluate(np.random.rand(100, 3))
