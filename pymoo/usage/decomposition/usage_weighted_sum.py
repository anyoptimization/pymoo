from pymoo.decomposition.weighted_sum import WeightedSum
from pymoo.factory import get_problem
import numpy as np

problem = get_problem("zdt1")

F = problem.evaluate(np.random.random((100, 30)))

val = WeightedSum().do(F, weights=np.array([[0.5, 0.5], [0.25, 0.25]]))

print(len(val))