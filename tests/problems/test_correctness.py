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


problems = [
    ('DTLZ1', [10, 3]), ('DTLZ2', [10, 3]), ('DTLZ3', [10, 3]), ('DTLZ4', [10, 3]), ('DTLZ5', [10, 3]),
    ('DTLZ6', [10, 3]), ('DTLZ7', [10, 3]),
    ('C1DTLZ1', [12, 3]), ('C2DTLZ2', [12, 3]), ('C3DTLZ1', [12, 3]), ('C3DTLZ4', [7, 3]),
    ('ZDT1', [10]), ('ZDT2', [10]), ('ZDT3', [10]), ('ZDT4', [10]), ('ZDT6', [10]),
    ('TNK', []), ('Rosenbrock', [10]), ('Rastrigin', [10]), ('Griewank', [10]), ('OSY', []), ('Kursawe', []),
    ('Welded_Beam', []), ('Carside', []), ('BNH', []), ('Cantilevered_Beam', []), ('Pressure_Vessel', []),
    ('G01', []), ('G02', []), ('G03', []), ('G04', []), ('G05', []), ('G06', []), ('G07', []), ('G08', []),
    ('G09', []), ('G10', []),
    # ('ctp1', []), ('ctp2', []), ('ctp3', []), ('ctp4', []), ('ctp5', []), ('ctp6', []), ('ctp7', []), ('ctp8', []),
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
            _F, _G, _CV, _dF, _dG = problem.evaluate(X, return_values_of=["F", "G", "CV", "dF", "dG"])

            if problem.n_obj == 1:
                F = F[:, None]

            self.assertTrue(anp.all(anp.abs(_F - F) < 0.00001))

            if problem.n_constr > 0:
                self.assertTrue(anp.all(anp.abs(_CV[:, 0] - CV) < 0.0001))


if __name__ == '__main__':
    unittest.main()
