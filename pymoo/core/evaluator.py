import numpy as np

from pymoo.core.individual import Individual
from pymoo.core.population import Population


class Evaluator:
    """

    The evaluator class which is used during the algorithm execution to limit the number of evaluations.
    This can be based on convergence, maximum number of evaluations, or other criteria.

    """

    def __init__(self,
                 skip_already_evaluated=True,
                 evaluate_values_of=["F", "G", "H"]):
        self.n_eval = 0
        self.evaluate_values_of = evaluate_values_of
        self.skip_already_evaluated = skip_already_evaluated

    def eval(self,
             problem,
             pop,
             skip_already_evaluated=None,
             evaluate_values_of=None,
             count_evals=True,
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

        if evaluate_values_of is None:
            evaluate_values_of = self.evaluate_values_of

        is_individual = isinstance(pop, Individual)
        is_numpy_array = isinstance(pop, np.ndarray) and not isinstance(pop, Population)

        # make sure the object is a population
        if is_individual or is_numpy_array:
            pop = Population().create(pop)

        # find indices to be evaluated
        if skip_already_evaluated or (skip_already_evaluated is None and self.skip_already_evaluated):
            I = []
            for i, individual in enumerate(pop):

                # if the evaluated has been overwritten to be none or not a set
                if not isinstance(individual.evaluated, set):
                    individual.evaluated = set()

                # if not everything has been evaluated
                if not all([e in individual.evaluated for e in evaluate_values_of]):
                    I.append(i)

        else:
            I = np.arange(len(pop))

        # update the function evaluation counter
        if count_evals:
            self.n_eval += len(I)

        # actually evaluate all solutions using the function that can be overwritten
        if len(I) > 0:
            self._eval(problem, pop[I], evaluate_values_of=evaluate_values_of, **kwargs)

        if is_individual:
            return pop[0]
        elif is_numpy_array:
            if len(pop) == 1:
                pop = pop[0]
            return tuple([pop.get(e) for e in self.evaluate_values_of])
        else:
            return pop

    def _eval(self, problem, pop, evaluate_values_of=None, **kwargs):
        evaluate_values_of = self.evaluate_values_of if evaluate_values_of is None else evaluate_values_of

        out = problem.evaluate(pop.get("X"),
                               return_values_of=evaluate_values_of,
                               return_as_dictionary=True,
                               **kwargs)

        for key, val in out.items():
            if val is None:
                continue
            else:
                pop.set(key, val)

        for ind in pop:
            ind.evaluated.update(out.keys())


class VoidEvaluator(Evaluator):

    def __init__(self, value=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def eval(self, problem, pop, **kwargs):
        val = self.value
        if val is not None:
            for individual in pop:
                if individual.F is None:
                    individual.F = np.full(problem.n_obj, val)
                    individual.G = np.full(problem.n_ieq_constr, val) if problem.n_ieq_constr > 0 else None
                    individual.H = np.full(problem.n_eq_constr, val) if problem.n_eq_constr else None
                    individual.CV = [-np.inf]
                    individual.feasible = [False]
