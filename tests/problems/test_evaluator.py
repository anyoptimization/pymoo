import unittest

import numpy as np

from pymoo.factory import get_problem
from pymoo.model.evaluator import Evaluator
from pymoo.model.individual import Individual
from pymoo.model.population import Population

problem = get_problem("Rastrigin")

X = np.random.random((100, problem.n_var))

F = problem.evaluate(X, return_values_of=["F"])

class EvaluatorTest(unittest.TestCase):

    def test_evaluate_array(self):
        evaluator = Evaluator(evaluate_values_of=["F", "CV"])
        _F, _CV = evaluator.eval(problem, X)
        np.testing.assert_allclose(F, _F)
        self.assertTrue(evaluator.n_eval == len(X))

    def test_evaluate_array_single(self):
        evaluator = Evaluator(evaluate_values_of=["F", "CV"])
        _F, _CV = evaluator.eval(problem, X[0])
        np.testing.assert_allclose(F[0], _F)
        self.assertTrue(evaluator.n_eval == 1)

    def test_evaluate_individual(self):
        evaluator = Evaluator()
        ind = evaluator.eval(problem, Individual(X=X[0]))
        np.testing.assert_allclose(F[0], ind.get("F"))
        self.assertTrue(evaluator.n_eval == 1)

    def test_evaluate_pop(self):
        evaluator = Evaluator()
        pop = Population().new("X", X)
        evaluator.eval(problem, pop)
        np.testing.assert_allclose(F, pop.get("F"))
        self.assertTrue(evaluator.n_eval == len(X))

    def test_preevaluated(self):
        evaluator = Evaluator()
        pop = Population().new("X", X)
        evaluator.eval(problem, pop)

        pop[range(30)].set("F", None)

        evaluator = Evaluator()
        evaluator.eval(problem, pop)

        np.testing.assert_allclose(F, pop.get("F"))
        self.assertTrue(evaluator.n_eval == 30)











if __name__ == '__main__':
    unittest.main()
