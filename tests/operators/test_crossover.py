import unittest

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_crossover, get_problem, get_sampling
from pymoo.optimize import minimize
import numpy as np
np.random.seed(123)


class CrossoverTest(unittest.TestCase):
    def test_single_crossover(self):
        problem = get_problem('zdt1')
        sampling = get_sampling('real_random')
        pop = sampling.do(problem, n_samples=3)

        crossover = get_crossover('real_sbx', eta=20)
        crossover.do(problem, pop[0], pop[1])
        crossover.do(problem, pop, np.array([[0, 1]]))


        crossover = get_crossover('real_de')
        crossover.do(problem, pop[0], pop[1], pop[2])
        crossover.do(problem, pop, np.array([[0, 1, 2]]))


    def test_crossover(self):
        for crossover in ['real_de', 'real_sbx', 'real_exp']:
            print(crossover)
            method = GA(pop_size=20, crossover=get_crossover(crossover, prob=0.95))
            minimize(get_problem("sphere"), method, ("n_gen", 20))

        for crossover in ['bin_ux', 'bin_hux', 'bin_one_point', 'bin_two_point']:
            print(crossover)
            method = NSGA2(pop_size=20, crossover=get_crossover(crossover, prob=0.95))
            minimize(get_problem("zdt5"), method, ("n_gen", 20))






if __name__ == '__main__':
    unittest.main()
