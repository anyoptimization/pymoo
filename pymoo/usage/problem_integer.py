import numpy as np
from pymop.factory import get_problem_from_func

from pymoo.operators.crossover.bin_uniform_crossover import BinaryUniformCrossover
from pymoo.operators.mutation.bin_bitflip_mutation import BinaryBitflipMutation
from pymoo.operators.sampling.bin_random_sampling import BinaryRandomSampling
from pymoo.optimize import minimize
from pymop import create_random_knapsack_problem, Problem


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2, n_obj=1, n_constr=1, xl=0, xu=10, type_var=np.double)

    def _evaluate(self, x, f, g, *args, **kwargs):
        f[:, 0] = - np.sum(np.power(x, 2), axis=1)
        g[:, 0] = 10 - 0.5 * x[:, 0] - x[:, 1]


problem = MyProblem()

res = minimize(problem,
               method='ga',
               method_args={
                   'pop_size': 100,
                   #'sampling': BinaryRandomSampling(),
                   #'crossover': BinaryUniformCrossover(),
                   #'mutation': BinaryBitflipMutation(),
                   'eliminate_duplicates': True,
               },
               termination=('n_gen', 20),
               disp=True)

print("Best solution found: %s" % res.X.astype(np.int))
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)
