import numpy as np

from pymoo.factory import get_problem
from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
from pymoo.core.population import Population

problem = get_problem("Rastrigin")

X = np.random.random((100, problem.n_var))

F = problem.evaluate(X, return_values_of=["F"])


def test_evaluate_array():
    evaluator = Evaluator(evaluate_values_of=["F", "CV"])
    _F, _CV = evaluator.eval(problem, X)
    np.testing.assert_allclose(F, _F)
    assert evaluator.n_eval == len(X)


def test_evaluate_array_single():
    evaluator = Evaluator(evaluate_values_of=["F", "CV"])
    _F, _CV = evaluator.eval(problem, X[0])
    np.testing.assert_allclose(F[0], _F)
    assert evaluator.n_eval == 1


def test_evaluate_individual():
    evaluator = Evaluator()
    ind = evaluator.eval(problem, Individual(X=X[0]))
    np.testing.assert_allclose(F[0], ind.get("F"))
    assert evaluator.n_eval == 1


def test_evaluate_pop():
    evaluator = Evaluator()
    pop = Population.new("X", X)
    evaluator.eval(problem, pop)
    np.testing.assert_allclose(F, pop.get("F"))
    assert evaluator.n_eval == len(X)


def test_preevaluated():
    evaluator = Evaluator()
    pop = Population.new("X", X)
    evaluator.eval(problem, pop)

    pop[range(30)].set("evaluated", None)

    evaluator = Evaluator()
    evaluator.eval(problem, pop)

    np.testing.assert_allclose(F, pop.get("F"))
    assert evaluator.n_eval == 30
