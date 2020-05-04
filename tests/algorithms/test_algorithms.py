import unittest

import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem, Problem, ZDT
from pymoo.optimize import minimize

class MyThreadedProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=1,
                         n_constr=0,
                         elementwise_evaluation=True,
                         parallelization=("threads", 4),
                         xl=np.array([0, 0]),
                         xu=np.array([100, 100]))

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x[0] + x[1]


class AlgorithmTest(unittest.TestCase):

    def test_same_seed_same_result(self):
        problem = get_problem("zdt3")
        algorithm = NSGA2(pop_size=100, eliminate_duplicates=True)

        res1 = minimize(problem, algorithm, ('n_gen', 20), seed=1)
        np.random.seed(200)
        res2 = minimize(problem, algorithm, ('n_gen', 20), seed=1)

        self.assertEqual(res1.X.shape, res2.X.shape)
        self.assertTrue(np.all(np.allclose(res1.X, res2.X)))

    def test_no_pareto_front_given(self):
        class ZDT1NoPF(ZDT):
            def _evaluate(self, x, out, *args, **kwargs):
                f1 = x[:, 0]
                g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:, 1:], axis=1)
                f2 = g * (1 - np.power((f1 / g), 0.5))
                out["F"] = np.column_stack([f1, f2])

        algorithm = NSGA2(pop_size=100, eliminate_duplicates=True)
        minimize(ZDT1NoPF(), algorithm, ('n_gen', 20), seed=1, verbose=True)

    def test_no_feasible_solution_found(self):
        class MyProblem(Problem):

            def __init__(self):
                super().__init__(n_var=2,
                                 n_obj=1,
                                 n_constr=36,
                                 xl=np.array([0, 0]),
                                 xu=np.array([100, 100]))

            def _evaluate(self, x, out, *args, **kwargs):
                f1 = x[:, 0] + x[:, 1]
                out["F"] = np.column_stack([f1])
                out["G"] = np.ones(len(x))

        res = minimize(MyProblem(),
                       NSGA2(),
                       ("n_gen", 10),
                       seed=1)

        self.assertEqual(res.X, None)
        self.assertEqual(res.F, None)
        self.assertEqual(res.G, None)

        res = minimize(MyProblem(),
                       NSGA2(),
                       ("n_gen", 10),
                       seed=1,
                       verbose=True,
                       return_least_infeasible=True,
                       save_history=True)

        self.assertAlmostEqual(res.CV[0], 1.0)


    def test_thread_pool(self):
        minimize(MyThreadedProblem(),
                 NSGA2(),
                 ("n_gen", 10),
                 seed=1,
                 save_history=False)


if __name__ == '__main__':
    unittest.main()
