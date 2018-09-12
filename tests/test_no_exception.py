import unittest

from pymoo.optimize import minimize
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.problems.osy import OSY
from pymop.problems.rastrigin import Rastrigin
from pymop.problems.zdt import ZDT1, ZDT4
from tests.test_problems import AlwaysInfeasibleProblem


class NoExceptionTest(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.problems = [ZDT1(n_var=30), ZDT4(n_var=5), Rastrigin(), OSY(), AlwaysInfeasibleProblem()]

        self.algorithms = []
        for name in ['nsga2', 'nsga3', 'unsga3']:
            d = {'name': name, 'pop_size': 100, 'n_eval': 300}
            self.algorithms.append(d)

    def test_no_exception(self):

        for problem in self.problems:

            for algorithm in self.algorithms:

                try:
                    minimize(problem,
                             method=algorithm['name'],
                             method_args={**algorithm, 'ref_dirs': UniformReferenceDirectionFactory(n_dim=problem.n_obj, n_points=100).do()},
                             termination=('n_eval', algorithm['n_eval']),
                             seed=2,
                             save_history=True,
                             disp=False)

                except:
                    print(problem)
                    print(algorithm)
                    self.fail("minimize() raised ExceptionType unexpectedly!")


if __name__ == '__main__':
    unittest.main()
