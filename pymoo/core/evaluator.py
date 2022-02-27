import numpy as np

from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.default import Default


class Evaluator:

    def __init__(self,
                 skip_already_evaluated: bool = True,
                 evaluate_values_of: list = ["F", "G", "H"],
                 tcv=Default.tcv,
                 individual=None):

        """
        The evaluator has the purpose to glue the problem with the population/individual objects.
        Additionally, it serves as a bookkeeper to store determine the number of function evaluations of runs, time,
        and others.


        Parameters
        ----------
        skip_already_evaluated : bool
            If individual that are already evaluated shall be skipped.

        evaluate_values_of : list
            The type of values to be asked the problem to evaluated. By default all objective, ieq. and eq. constraints.

        tcv : Total Constraint Violation
            The class to calculate the constraint violation of an object.

        individual : Individual
            The type of individual to be used in the population after evaluation. If `None` the default type is used.

        """

        self.evaluate_values_of = evaluate_values_of
        self.skip_already_evaluated = skip_already_evaluated
        self.tcv = tcv
        self.individual = individual

        # current number of function evaluations - initialized to zero
        self.n_eval = 0

    def eval(self,
             problem: Problem,
             pop: Population,
             skip_already_evaluated: bool = None,
             evaluate_values_of: list = None,
             count_evals: bool = True,
             **kwargs):

        # load the default settings from the evaluator object if not already provided
        evaluate_values_of = self.evaluate_values_of if evaluate_values_of is None else evaluate_values_of
        skip_already_evaluated = self.skip_already_evaluated if skip_already_evaluated is None else skip_already_evaluated

        # check the type of the input
        is_individual = isinstance(pop, Individual)
        is_numpy_array = isinstance(pop, np.ndarray) and not isinstance(pop, Population)

        # make sure the object is a population
        if is_individual or is_numpy_array:
            pop = Population().create(pop)

        # find indices that will be evaluated
        if skip_already_evaluated:

            # the indices to be evaluated
            I = []

            # iterate over all individuals
            for i, individual in enumerate(pop):

                # if the evaluated has been overwritten to be none or not a set
                if not isinstance(individual.evaluated, set):
                    individual.evaluated = set()

                # if not everything has been evaluated
                if not all([e in individual.evaluated for e in evaluate_values_of]):
                    I.append(i)

        # if skipping is deactivated simply make the index being all individuals
        else:
            I = np.arange(len(pop))

        # evaluate the solutions (if there are any)
        if len(I) > 0:

            # check for the individual class - if a different one should be used do the conversion
            for i in I:
                if self.individual is not None:
                    if not isinstance(pop[i], self.individual.__class__):
                        pop[i] = self.individual.copy(pop[i])

            # do the actual evaluation - call the sub-function to set the corresponding values to the population
            self._eval(problem, pop[I], evaluate_values_of, **kwargs)

        # update the function evaluation counter
        if count_evals:
            self.n_eval += len(I)

        if is_individual:
            return pop[0]
        elif is_numpy_array:
            if len(pop) == 1:
                pop = pop[0]
            return tuple([pop.get(e) for e in self.evaluate_values_of])
        else:
            return pop

    def _eval(self, problem, pop, evaluate_values_of, **kwargs):

        out = problem.evaluate(pop.get("X"),
                               return_values_of=evaluate_values_of,
                               return_as_dictionary=True,
                               **kwargs)

        # for each of the attributes set it too the problem
        for key, val in out.items():
            if val is None:
                continue
            else:
                pop.set(key, val)

        # finally set all the attributes to be evaluated for all individuals
        for ind in pop:
            ind.evaluated.update(out.keys())

        # directly calculate the tcv and store the values in the population
        if self.tcv is not None:
            self.tcv.do(pop, inplace=True)


class VoidEvaluator(Evaluator):

    def __init__(self, value=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def eval(self, problem, pop, **kwargs):
        val = self.value
        if val is not None:
            for individual in pop:
                if len(individual.evaluated) == 0:
                    individual.F = np.full(problem.n_obj, val)
                    individual.G = np.full(problem.n_ieq_constr, val) if problem.n_ieq_constr > 0 else None
                    individual.H = np.full(problem.n_eq_constr, val) if problem.n_eq_constr else None
                    individual.CV = [-np.inf]
                    individual.feas = [False]
