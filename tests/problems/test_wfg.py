import unittest

import autograd.numpy as anp

from pymoo.factory import get_problem
import os


def load(name):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")

    X = anp.loadtxt(os.path.join(path, "%s.x" % name))

    try:
        F = anp.loadtxt(os.path.join(path, "%s.f" % name))

        CV = None
        if os.path.exists(os.path.join(path, "%s.cv" % name)):
            CV = anp.loadtxt(os.path.join(path, "%s.cv" % name))

    except:
        return X, None, None

    return X, F, CV

# Uncomment to check WFG problems
problems = [
    # ('WFG1', [2,4,2]),
    # ('WFG2', [2,4,2]),
    # ('WFG3', [2,4,2]),
    # ('WFG4', [2,4,2]),
    # ('WFG5', [2,4,2]),
    # ('WFG6', [2,4,2]),
    # ('WFG7', [2,4,2]),
    # ('WFG8', [2,4,2]),
    ('WFG9', [2,4,2])
]



class CorrectnessTest(unittest.TestCase):

    def test_problems(self):
        for entry in problems:
            name, params = entry
            print("Testing: " + name)

            X, F, CV = load(name)

            if F is None:
                print("Warning: No correctness check for %s" % name)
                continue

            problem = get_problem(name, *params)
            _F, _G, _CV = problem.evaluate(X, return_values_of=["F", "G", "CV"])

            if problem.n_obj == 1:
                F = F[:, None]

            self.assertTrue(anp.all(anp.abs(_F - F) < 0.00001))

            if problem.n_constr > 0:
                self.assertTrue(anp.all(anp.abs(_CV[:, 0] - CV) < 0.0001))


if __name__ == '__main__':
    unittest.main()
