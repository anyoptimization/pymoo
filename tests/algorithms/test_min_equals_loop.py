import unittest

import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem, get_termination
from pymoo.optimize import minimize


class AlgorithmTest(unittest.TestCase):

    def test_same_result(self):
        problem = get_problem("zdt1")

        algorithm = NSGA2(pop_size=100)
        min_res = minimize(problem, algorithm, ('n_gen', 200), seed=1, verbose=True)

        algorithm = NSGA2(pop_size=100)
        algorithm.setup(problem, ('n_gen', 200), seed=1)
        while algorithm.has_next():
            algorithm.next()
            print(algorithm.n_gen)
        loop_res = algorithm.result()

        np.testing.assert_allclose(min_res.X, loop_res.X)
