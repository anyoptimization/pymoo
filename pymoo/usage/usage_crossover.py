import numpy as np

from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.problems.single import Rastrigin

problem = Rastrigin(n_var=30)
crossover = SimulatedBinaryCrossover(eta=20)

pop = FloatRandomSampling().do(problem, 2)

parents = np.array([[0, 1]])

off = crossover.do(problem, pop, parents)

print(off)

ind_a = pop[0]
ind_b = pop[1]

off = crossover.do(problem, ind_a, ind_b)
print(off)