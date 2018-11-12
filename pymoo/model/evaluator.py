import numpy as np

from pymoo.model.individual import Individual
from pymoo.model.population import Population


class Evaluator:
    """

    The evaluator class which is used during the algorithm execution to limit the number of evaluations.
    This can be based on convergence, maximum number of evaluations, or other criteria.

    """

    def __init__(self):
        self.n_eval = 0

    def eval(self, problem, X, **kwargs):
        """

        This function is used to return the result of one valid evaluation.

        Parameters
        ----------
        problem : class
            The problem which is used to be evaluated
        X : np.array or Population object
        kwargs : dict
            Additional arguments which might be necessary for the problem to evaluate.

        """

        if isinstance(X, Individual):

            self.n_eval += 1
            X.F, X.CV, X.G = problem.evaluate(X.X,
                                        return_constraint_violation=True,
                                        return_constraints=True,
                                        individuals=X,
                                        **kwargs)
            X.feasible = X.CV <= 0

        elif isinstance(X, Population):

            pop, _X = X, X.get("X")
            self.n_eval += len(pop)

            F, CV, G = problem.evaluate(_X,
                                        return_constraint_violation=True,
                                        return_constraints=True,
                                        individuals=pop,
                                        **kwargs)
            pop.set("F", F, "CV", CV, "G", G, "feasible", (CV <= 0))

        elif isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                self.n_eval += 1
            else:
                self.n_eval += X.shape[0]
            return problem.evaluate(X, **kwargs)
