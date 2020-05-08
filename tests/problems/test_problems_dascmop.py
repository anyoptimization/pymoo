import os
import unittest

import numpy as np

from pymoo.factory import get_problem
from pymoo.problems.multi import DIFFICULTIES


def load(name, diff):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "DASCMOP", str(diff))
    X = np.loadtxt(os.path.join(path, "%s.x" % name))
    F = np.loadtxt(os.path.join(path, "%s.f" % name))
    CV = np.loadtxt(os.path.join(path, "%s.cv" % name))[:, None]
    return X, F, -CV


class CorrectnessTest(unittest.TestCase):

    def test_problems(self):

        for k, diff_factors in enumerate(DIFFICULTIES):

            problems = [
                get_problem("dascmop1", diff_factors),
                get_problem("dascmop2", diff_factors),
                get_problem("dascmop3", diff_factors),
                get_problem("dascmop4", diff_factors),
                get_problem("dascmop5", diff_factors),
                get_problem("dascmop6", diff_factors),
                get_problem("dascmop7", diff_factors),
                get_problem("dascmop8", diff_factors),
                get_problem("dascmop9", diff_factors)
            ]

            for problem in problems:
                name = str(problem.__class__.__name__)
                print("Testing: " + name, "diff:", k)

                X, F, CV = load(name, k)
                _F, _CV = problem.evaluate(X, return_values_of=["F", "CV"])

                np.testing.assert_allclose(_F, F)
                np.testing.assert_allclose(_CV, CV)


if __name__ == '__main__':
    unittest.main()
