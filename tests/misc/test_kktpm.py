import os
import unittest

import numpy as np

from pymoo.factory import get_problem
from pymoo.performance_indicator.kktpm import KKTPM
from tests import path_to_test_resources

folder = path_to_test_resources("kktpm")


class KKTPMTest(unittest.TestCase):

    def test_correctness(self):

        np.random.seed(1)

        setup = {
            "bnh": {'utopian_epsilon': 0.0, "ideal_point": np.array([-0.05, -0.05]), "rho": 0.0},
            "zdt1": {'utopian_epsilon': 1e-3},
            "zdt2": {'utopian_epsilon': 1e-4},
            "zdt3": {'utopian_epsilon': 1e-4, "ideal_point": np.array([0.0, -1.0])},
            "osy": {'utopian_epsilon': 0.0, "ideal_point": np.array([-300, -0.05]), "rho": 0.0}
        }

        for str_problem, params in setup.items():

            problem = get_problem(str_problem)

            import os

            def load_file(f):
                return np.loadtxt(os.path.join(folder, "%s_%s.txt" % (str_problem, f)))

            X = load_file("x")
            _F, _G, _dF, _dG = problem.evaluate(X, return_values_of=["F", "G", "dF", "dG"])
            F, G, dF, dG = load_file("f"), load_file("g"), load_file("df"), load_file("dg")

            dF = dF.reshape(_dF.shape)
            np.testing.assert_almost_equal(F, _F, decimal=5)
            np.testing.assert_almost_equal(dF, _dF, decimal=5)

            if problem.n_constr > 0:
                G = G[:, :problem.n_constr]
                dG = dG[:, :problem.n_constr * problem.n_var].reshape(_dG.shape)

                np.testing.assert_almost_equal(G, _G, decimal=5)
                np.testing.assert_almost_equal(dG, _dG, decimal=5)

            # indices = np.random.permutation(X.shape[0])[:100]
            indices = np.arange(X.shape[0])

            # load the correct results
            kktpm = load_file("kktpm")[indices]

            # calculate the KKTPM measure
            # _kktpm, _ = KKTPM(var_bounds_as_constraints=True).calc(np.array([[4.8, 3.0]]), problem, **params)
            # _kktpm, _ = KKTPM(var_bounds_as_constraints=True).calc(X[[55]], problem, rho=0, **params)
            _kktpm  = KKTPM(var_bounds_as_constraints=True).calc(X[indices], problem, **params)
            error = np.abs(_kktpm[:, 0] - kktpm)

            for i in range(len(error)):

                if error[i] > 0.0001:
                    print("Error for ", str_problem)
                    print("index: ", i)
                    print("Error: ", error[i])
                    print("X", ",".join(np.char.mod('%f', X[i])))
                    print("Python: ", _kktpm[i])
                    print("Correct: ", kktpm[i])

                    import os
                    # os._exit(1)

            # make sure the results are almost equal
            np.testing.assert_almost_equal(kktpm, _kktpm[:, 0], decimal=4)
            print(str_problem, error.mean())


if __name__ == '__main__':
    unittest.main()
