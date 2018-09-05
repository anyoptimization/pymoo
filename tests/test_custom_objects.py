import unittest

import numpy as np

from pymoo.model.individual import Individual
from pymoo.model.sampling import Sampling
from pymoo.optimize import minimize
from pymop.problems.zdt import ZDT1, ZDT


class CustomObjectTest(unittest.TestCase):

    def test_custom_object_sampled_dtype_float(self):

        problem = ZDT1()

        class CustomObject(Individual):

            def __init__(self) -> None:
                super().__init__()
                self.stress = 0.0
                self.cr = 0

        class CustomObjectFloatRandomSampling(Sampling):
            def sample(self, problem, n_samples, **kwargs):
                l = []
                for i in range(n_samples):
                    obj = CustomObject()
                    obj.X = np.random.random(problem.n_var)
                    l.append(obj)
                return np.array(l, dtype=object)[:, None]

        try:
            minimize(problem,
                     method="nsga2",
                     method_args={'pop_size': 100, 'sampling': CustomObjectFloatRandomSampling()},
                     termination=('n_eval', 500),
                     seed=2,
                     save_history=False,
                     disp=False)

        except:
            self.fail("minimize() raised ExceptionType unexpectedly!")

    def test_no_custom_object_but_attributes_are_set_during_eval(self):

        class CustomZDT(ZDT):
            def __init__(self, n_var=30):
                ZDT.__init__(self, n_var)

            def _calc_pareto_front(self):
                x1 = np.arange(0, 1.01, 0.01)
                return np.array([x1, 1 - np.sqrt(x1)]).T

            def _evaluate(self, x, f, individuals):
                f[:, 0] = x[:, 0]
                g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:, 1:], axis=1)
                f[:, 1] = g * (1 - np.power((f[:, 0] / g), 0.5))
                for i in range(x.shape[0]):
                    individuals[i, 0].D['f1'] = f[i, 1]

        problem = CustomZDT()

        try:
            res = minimize(problem,
                           method="nsga2",
                           method_args={'pop_size': 100},
                           termination=('n_eval', 500),
                           seed=2,
                           save_history=False,
                           disp=False)

            for i in range(res['individuals'].shape[0]):
                self.assertTrue(res['individuals'][i, 0].D['f1'] == res['F'][i, 1])

        except:
            self.fail("minimize() raised ExceptionType unexpectedly!")


if __name__ == '__main__':
    unittest.main()
