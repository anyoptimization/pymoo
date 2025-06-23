import numpy as np

import pymoo.gradient
from pymoo.gradient.automatic import AutomaticDifferentiation
from pymoo.problems.multi import ZDT1

pymoo.gradient.activate("autograd.numpy")

problem = ZDT1()

problem = AutomaticDifferentiation(problem)

X = np.random.random((100, problem.n_var))

F, dF = problem.evaluate(X, return_values_of=["F", "dF"])
