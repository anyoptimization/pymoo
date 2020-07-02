import os
import unittest

import numpy as np

from pymoo.factory import get_problem


def load(name):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")
    X = np.loadtxt(os.path.join(path, "%s.x" % name))
    F = np.loadtxt(os.path.join(path, "%s.f" % name))
    CV = np.loadtxt(os.path.join(path, "%s.cv" % name))[:, None]
    return X, F, CV


class CorrectnessTest(unittest.TestCase):

    def test_problems(self):
        problems = [
            get_problem("dc1dtlz1"),
            get_problem("dc1dtlz3"),
            get_problem("dc2dtlz1"),
            get_problem("dc2dtlz3"),
            get_problem("dc3dtlz1"),
            get_problem("dc3dtlz3"),
        ]

        for problem in problems:
            name = str(problem.__class__.__name__)
            print("Testing: " + name)

            X, F, CV = load(name)
            _F, _CV, _G = problem.evaluate(X, return_values_of=["F", "CV", "G"])

            if _G.shape[1] > 1:
                # We need to do a special CV calculation to test for correctness since
                # the original code does not sum the violations but takes the maximum
                _CV = np.max(_G, axis=1)[:, None]
                _CV = np.maximum(_CV, 0)

            np.testing.assert_allclose(_F, F)
            np.testing.assert_allclose(_CV, CV)



if __name__ == '__main__':
    unittest.main()


