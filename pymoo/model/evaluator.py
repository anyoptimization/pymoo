import numpy as np

from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.model.problem import Problem
from pymoo.util.misc import at_least_2d_array


def set_feasibility(pop):
    for ind in pop:
        cv = ind.get("CV")
        if cv is not None:
            ind.feasible = cv <= 0


def set_cv(pop, feasbility=True):
    for ind in pop:
        if ind.G is None:
            ind.CV = np.zeros(1)
        else:
            ind.CV = Problem.calc_constraint_violation(at_least_2d_array(ind.G))[0]

    if feasbility:
        set_feasibility(pop)


class Evaluator:
    """

    The evaluator class which is used during the algorithm execution to limit the number of evaluations.
    This can be based on convergence, maximum number of evaluations, or other criteria.

    """

    def __init__(self,
                 skip_already_evaluated=True,
                 evaluate_values_of=["F", "CV", "G"]):
        self.n_eval = 0
        self.evaluate_values_of = evaluate_values_of
        self.skip_already_evaluated = skip_already_evaluated

    def eval(self,
             problem,
             pop,
             **kwargs):
        """

        This function is used to return the result of one valid evaluation.

        Parameters
        ----------
        problem : class
            The problem which is used to be evaluated
        pop : np.array or Population object
        kwargs : dict
            Additional arguments which might be necessary for the problem to evaluate.

        """

        is_individual = isinstance(pop, Individual)
        is_numpy_array = isinstance(pop, np.ndarray) and not isinstance(pop, Population)

        # make sure the object is a population
        if is_individual or is_numpy_array:
            pop = Population().create(pop)

        # find indices to be evaluated
        if self.skip_already_evaluated:
            I = [k for k in range(len(pop)) if pop[k].F is None]
        else:
            I = np.arange(len(pop))

        # update the function evaluation counter
        self.n_eval += len(I)

        # actually evaluate all solutions using the function that can be overwritten
        if len(I) > 0:
            self._eval(problem, pop[I], **kwargs)

            # set the feasibility attribute if cv exists
            set_feasibility(pop[I])

        if is_individual:
            return pop[0]
        elif is_numpy_array:
            if len(pop) == 1:
                pop = pop[0]
            return tuple([pop.get(e) for e in self.evaluate_values_of])
        else:
            return pop

    def _eval(self, problem, pop, **kwargs):

        out = problem.evaluate(pop.get("X"),
                               return_values_of=self.evaluate_values_of,
                               return_as_dictionary=True,
                               **kwargs)

        for key, val in out.items():
            if val is None:
                continue
            else:
                pop.set(key, val)
