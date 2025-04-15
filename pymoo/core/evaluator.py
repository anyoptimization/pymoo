import numpy as np

from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.problem import Problem


class Evaluator:

    def __init__(self,
                 skip_already_evaluated: bool = True,
                 evaluate_values_of: list = ["F", "G", "H"],
                 callback=None):

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

        """

        self.evaluate_values_of = evaluate_values_of
        self.skip_already_evaluated = skip_already_evaluated
        self.callback = callback

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

        # make sure the object is a population
        if is_individual:
            pop = Population().create(pop)

        # filter the index to have individual where not all attributes have been evaluated
        if skip_already_evaluated:
            I = [i for i, ind in enumerate(pop) if not all([e in ind.evaluated for e in evaluate_values_of])]

        # if skipping is deactivated simply make the index being all individuals
        else:
            I = np.arange(len(pop))

        # evaluate the solutions (if there are any)
        if len(I) > 0:

            # do the actual evaluation - call the sub-function to set the corresponding values to the population
            self._eval(problem, pop[I], evaluate_values_of, **kwargs)

        # update the function evaluation counter
        if count_evals:
            self.n_eval += len(I)

        # allow to have a callback registered
        if self.callback:
            self.callback(pop)

        if is_individual:
            return pop[0]
        else:
            return pop

    def _eval(self, problem, pop, evaluate_values_of, **kwargs):

        # get the design space value from the individuals
        X = pop.get("X")

        # call the problem to evaluate the solutions
        out = problem.evaluate(X, return_values_of=evaluate_values_of, return_as_dictionary=True, **kwargs)

        # for each of the attributes set it to the problem
        for key, val in out.items():
            if val is not None:
                pop.set(key, val)

        # finally set all the attributes to be evaluated for all individuals
        pop.apply(lambda ind: ind.evaluated.update(out.keys()))


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
