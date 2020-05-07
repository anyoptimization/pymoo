import os
import unittest

import numpy as np

from pymoo.factory import get_problem


def load(name):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "MW")
    X = np.loadtxt(os.path.join(path, "%s.x" % name))
    F = np.loadtxt(os.path.join(path, "%s.f" % name))
    CV = np.loadtxt(os.path.join(path, "%s.cv" % name))[:, None]
    return X, F, CV


class CorrectnessTest(unittest.TestCase):

    def test_problems(self):
        problems = [
            get_problem("mw1"),
            get_problem("mw2"),
            get_problem("mw3"),
            get_problem("mw4"),
            get_problem("mw5"),
            get_problem("mw6"),
            get_problem("mw7"),
            get_problem("mw8"),
            get_problem("mw9"),
            get_problem("mw10"),
            get_problem("mw11"),
            get_problem("mw12"),
            get_problem("mw13"),
            get_problem("mw14"),
        ]

        for problem in problems:
            name = str(problem.__class__.__name__)
            print("Testing: " + name)

            X, F, CV = load(name)
            _F, _CV = problem.evaluate(X, return_values_of=["F", "CV"])

            np.testing.assert_allclose(_F, F)
            np.testing.assert_allclose(_CV, CV)



if __name__ == '__main__':
    unittest.main()


