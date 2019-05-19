import numpy as np

from pymoo.factory import get_problem

problem = get_problem("zdt1")

F, dF, CV = problem.evaluate(np.random.random((100, 30)), return_values_of=["F", "dF", "CV"])
