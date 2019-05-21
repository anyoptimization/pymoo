import unittest

import numpy as np

from pymoo.performance_indicator.kktpm import KKTPM
from pymoo.factory import get_problem


class KKTPMTest(unittest.TestCase):

    def test_correctness(self):

        np.random.seed(1)

        setup = {
            "zdt1": {'utopian_epsilon': 1e-3},
            "zdt2": {'utopian_epsilon': 1e-4},
            "zdt3": {'utopian_epsilon': 1e-4, "ideal_point": np.array([0.0, -1.0])},
            "osy": {'utopian_epsilon': 0.0, "ideal_point": np.array([-300, -0.05])}
        }

        for str_problem, parameter in setup.items():

            problem = get_problem(str_problem)

            import os
            X = np.loadtxt(os.path.join("resources", "kktpm", "%s.x" % str_problem), delimiter=",")
            #F = np.loadtxt(os.path.join("resources", "kktpm", "%s.f" % str_problem))
            #G = np.loadtxt(os.path.join("resources", "kktpm", "%s.g" % str_problem))

            _F, _G, _dF, _dG = problem.evaluate(X, return_values_of=["F", "G", "dF", "dG"])

            #self.assertTrue(np.abs(F - _F).mean() < 1e-6)
            #self.assertTrue(np.abs(G - _G).mean() < 1e-6)

            #indices = np.random.permutation(X.shape[0])[:100]
            indices = np.arange(X.shape[0])

            # load the correct results
            correct = np.loadtxt(os.path.join("resources", "kktpm", "%s.kktpm" % str_problem))[indices]

            # calculate the KKTPM measure
            kktpm, _ = KKTPM(var_bounds_as_constraints=True).calc(X[indices], problem, **parameter)
            error = np.abs(kktpm[:, 0] - correct)

            for i in range(len(error)):

                if error[i] > 0.0001:
                    print("Error for ", str_problem)
                    print("index: ", i)
                    print("Error: ", error[i])
                    print("X", ",".join(np.char.mod('%f', X[i])))
                    print("Python: ", kktpm[i])
                    print("Correct: ", correct[i])

                    import os
                    os._exit(1)

            # make sure the results are almost equal
            self.assertTrue(error.mean() < 1e-6)

            print(str_problem, error.mean())


if __name__ == '__main__':
    unittest.main()


