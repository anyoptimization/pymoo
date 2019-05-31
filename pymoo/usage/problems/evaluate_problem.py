import numpy as np

from pymoo.problems.multi.zdt import ZDT1

problem = ZDT1(n_var=10)

# if the function does not have any constraints only function values are returned
F = problem.evaluate(np.random.random(10))

# in case more than one solution should be evaluated you can provide a matrix
F = problem.evaluate(np.random.random((100, 10)))

from pymoo.problems.multi.welded_beam import WeldedBeam
problem = WeldedBeam()

# by default a problem with constrained will also return the constraint violation
F, CV = problem.evaluate(np.random.random((100, 4)))

# if only specific values are required return_values_of can be defined
F = problem.evaluate(np.random.random((100, 4)), return_values_of=["F"])

# in this case more values are returned (also the gradient of the objective values!)
F, G, CV, dF = problem.evaluate(np.random.random((100, 4)), return_values_of=["F", "G", "CV", "dF"])
