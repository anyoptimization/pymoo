import numpy as np

from pymoo.problems import get_problem
from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
from pymoo.core.population import Population

problem = get_problem("Rastrigin")

X = np.random.random((100, problem.n_var))

F = problem.evaluate(X)


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

    for ind in pop[:30]:
        ind.set("evaluated", set())

    evaluator = Evaluator()
    evaluator.eval(problem, pop)

    np.testing.assert_allclose(F, pop.get("F"))
    assert evaluator.n_eval == 30
