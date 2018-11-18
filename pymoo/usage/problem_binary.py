import numpy as np

from pymoo.operators.crossover.uniform_crossover import BinaryUniformCrossover
from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation
from pymoo.operators.sampling.bin_random_sampling import BinaryRandomSampling
from pymoo.optimize import minimize
from pymop import create_random_knapsack_problem

problem = create_random_knapsack_problem(30)
problem.type_var = np.bool

res = minimize(problem,
               method='ga',
               method_args={
                   'pop_size': 100,
                   'sampling': BinaryRandomSampling(),
                   'crossover': BinaryUniformCrossover(),
                   'mutation': BinaryBitflipMutation(),
                   'eliminate_duplicates': True,
               },
               termination=('n_gen', 100),
               disp=True)

print("Best solution found: %s" % res.X.astype(np.int))
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)
