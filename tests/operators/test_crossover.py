import unittest

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_crossover, get_problem
from pymoo.optimize import minimize


class CrossoverTest(unittest.TestCase):

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
