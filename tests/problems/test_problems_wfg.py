import os
import unittest

import autograd.numpy as anp

from pymoo.factory import get_problem, WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9


def load(name, n_obj):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "WFG", "%sobj" % n_obj)

    X = anp.loadtxt(os.path.join(path, "%s.x" % name))

    try:
        F = anp.loadtxt(os.path.join(path, "%s.f" % name))

        CV = None
        if os.path.exists(os.path.join(path, "%s.cv" % name)):
            CV = anp.loadtxt(os.path.join(path, "%s.cv" % name))

    except:
        return X, None, None

    return X, F, CV




class CorrectnessTest(unittest.TestCase):

    def test_problems(self):

        for n_obj, n_var, k in [(2, 6, 4), (3, 6, 4), (10, 20, 18)]:

            problems = [
                get_problem("wfg1", n_var, n_obj, k),
                get_problem("wfg2", n_var, n_obj, k),
                get_problem("wfg3", n_var, n_obj, k),
                get_problem("wfg4", n_var, n_obj, k),
                get_problem("wfg5", n_var, n_obj, k),
                get_problem("wfg6", n_var, n_obj, k),
                get_problem("wfg7", n_var, n_obj, k),
                get_problem("wfg8", n_var, n_obj, k),
                get_problem("wfg9", n_var, n_obj, k)
            ]

            for problem in problems:
                name = str(problem.__class__.__name__)
                print("Testing: " + name + "-" + str(n_obj))

                X, F, CV = load(name, n_obj)

                # other = from_optproblems(problem)
                # F = np.row_stack([other.objective_function(x) for x in X])

                if F is None:
                    print("Warning: No correctness check for %s" % name)
                    continue

                _F, _G, _CV = problem.evaluate(X, return_values_of=["F", "G", "CV"])

                if problem.n_obj == 1:
                    F = F[:, None]

                self.assertTrue(anp.all(anp.abs(_F - F) < 0.00001))

                if problem.n_constr > 0:
                    self.assertTrue(anp.all(anp.abs(_CV[:, 0] - CV) < 0.0001))



def from_optproblems(wfg):
    from optproblems.wfg import WFG1 as WFG1opt
    from optproblems.wfg import WFG2 as WFG2opt
    from optproblems.wfg import WFG3 as WFG3opt
    from optproblems.wfg import WFG4 as WFG4opt
    from optproblems.wfg import WFG5 as WFG5opt
    from optproblems.wfg import WFG6 as WFG6opt
    from optproblems.wfg import WFG7 as WFG7opt
    from optproblems.wfg import WFG8 as WFG8opt
    from optproblems.wfg import WFG9 as WFG9opt

    if isinstance(wfg, WFG1):
        return WFG1opt(wfg.n_obj, wfg.n_var, wfg.k)
    elif isinstance(wfg, WFG2):
        return WFG2opt(wfg.n_obj, wfg.n_var, wfg.k)
    elif isinstance(wfg, WFG3):
        return WFG3opt(wfg.n_obj, wfg.n_var, wfg.k)
    elif isinstance(wfg, WFG4):
        return WFG4opt(wfg.n_obj, wfg.n_var, wfg.k)
    elif isinstance(wfg, WFG5):
        return WFG5opt(wfg.n_obj, wfg.n_var, wfg.k)
    elif isinstance(wfg, WFG6):
        return WFG6opt(wfg.n_obj, wfg.n_var, wfg.k)
    elif isinstance(wfg, WFG7):
        return WFG7opt(wfg.n_obj, wfg.n_var, wfg.k)
    elif isinstance(wfg, WFG8):
        return WFG8opt(wfg.n_obj, wfg.n_var, wfg.k)
    elif isinstance(wfg, WFG9):
        return WFG9opt(wfg.n_obj, wfg.n_var, wfg.k)


if __name__ == '__main__':
    unittest.main()


