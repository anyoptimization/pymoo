import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT


def test_same_seed_same_result():
    problem = get_problem("zdt3")
    algorithm = NSGA2(pop_size=100, eliminate_duplicates=True)

    res1 = minimize(problem, algorithm, ('n_gen', 20), seed=1)
    np.random.seed(200)
    res2 = minimize(problem, algorithm, ('n_gen', 20), seed=1)

    np.testing.assert_almost_equal(res1.X, res2.X)
    np.testing.assert_almost_equal(res1.F, res2.F)


def test_no_pareto_front_given():
    class ZDT1NoPF(ZDT):
        def _evaluate(self, x, out, *args, **kwargs):
            f1 = x[:, 0]
            g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:, 1:], axis=1)
            f2 = g * (1 - np.power((f1 / g), 0.5))
            out["F"] = np.column_stack([f1, f2])

    algorithm = NSGA2(pop_size=100, eliminate_duplicates=True)
    minimize(ZDT1NoPF(), algorithm, ('n_gen', 20), seed=1, verbose=True)
    assert True


def test_no_feasible_solution_found():
    class MyProblem(Problem):

        def __init__(self):
            super().__init__(n_var=2,
                             n_obj=1,
                             n_ieq_constr=1,
                             xl=[0, 0],
                             xu=[100, 100])

        def _evaluate(self, x, out, *args, **kwargs):
            f1 = x[:, 0] + x[:, 1]
            out["F"] = np.column_stack([f1])
            out["G"] = np.ones(len(x))

    res = minimize(MyProblem(),
                   NSGA2(),
                   ("n_gen", 10),
                   seed=1)

    assert res.X is None
    assert res.F is None
    assert res.G is None

    res = minimize(MyProblem(),
                   NSGA2(),
                   ("n_gen", 10),
                   seed=1,
                   verbose=True,
                   return_least_infeasible=True,
                   save_history=True)

    assert res.CV[0] == 1.0


def test_thread_pool():
    class MyThreadedProblem(ElementwiseProblem):

        def __init__(self):
            super().__init__(n_var=2,
                             n_obj=1,
                             xl=np.array([0, 0]),
                             xu=np.array([100, 100]))

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = x[0] + x[1]

    minimize(MyThreadedProblem(),
             NSGA2(),
             ("n_gen", 10),
             seed=1,
             save_history=False)


def test_min_vs_loop_vs_infill():
    problem = get_problem("zdt1")
    n_gen = 30

    algorithm = NSGA2(pop_size=100)
    min_res = minimize(problem, algorithm, ('n_gen', n_gen), seed=1)

    algorithm = NSGA2(pop_size=100)
    algorithm.setup(problem, termination=('n_gen', n_gen), seed=1)
    while algorithm.has_next():
        algorithm.next()
    algorithm.finalize()
    loop_res = algorithm.result()

    np.testing.assert_allclose(min_res.X, loop_res.X)

    algorithm = NSGA2(pop_size=100)
    algorithm.setup(problem, termination=('n_gen', n_gen), seed=1)
    while algorithm.has_next():
        infills = algorithm.infill()
        Evaluator().eval(problem, infills)
        algorithm.advance(infills=infills)
    algorithm.finalize()
    infill_res = algorithm.result()

    np.testing.assert_allclose(min_res.X, infill_res.X)
