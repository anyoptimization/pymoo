"""Evaluation infrastructure for connecting problems and populations."""

from typing import Any

import numpy as np

from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.problem import Problem


class Evaluator:
    def __init__(
        self,
        skip_already_evaluated: bool = True,
        evaluate_values_of: list | None = None,
        callback=None,
    ):
        """Glue problem evaluation with population/individual objects.

        Additionally, serves as a bookkeeper to store the number of function evaluations,
        time, and other metadata.

        Args:
            skip_already_evaluated: If individuals that are already evaluated shall be skipped.
            evaluate_values_of: Types of values to be evaluated by the problem.
                Defaults to all objective, inequality and equality constraints
                (``["F", "G", "H"]``) when ``None``.
            callback: Optional callback function to execute after evaluation.
        """
        self.evaluate_values_of = (
            ["F", "G", "H"] if evaluate_values_of is None else evaluate_values_of
        )
        self.skip_already_evaluated = skip_already_evaluated
        self.callback = callback

        # current number of function evaluations - initialized to zero
        self.n_eval = 0

    def eval(
        self,
        problem: Problem,
        pop: Population,
        skip_already_evaluated: bool | None = None,
        evaluate_values_of: list[Any] | None = None,
        count_evals: bool = True,
        **kwargs,
    ):

        # load the default settings from the evaluator object if not already provided
        evaluate_values_of = (
            self.evaluate_values_of
            if evaluate_values_of is None
            else evaluate_values_of
        )
        skip_already_evaluated = (
            self.skip_already_evaluated
            if skip_already_evaluated is None
            else skip_already_evaluated
        )

        # check the type of the input
        is_individual = isinstance(pop, Individual)

        # make sure the object is a population
        if is_individual:
            pop = Population().create(pop)

        # filter the index to have individual where not all attributes have been evaluated
        if skip_already_evaluated:
            I = np.array(  # noqa: E741
                [
                    i
                    for i, ind in enumerate(pop)
                    if not all([e in ind.evaluated for e in evaluate_values_of])
                ]
            )

        # if skipping is deactivated simply make the index being all individuals
        else:
            I = np.arange(len(pop))  # noqa: E741

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
        out = problem.evaluate(
            X, return_values_of=evaluate_values_of, return_as_dictionary=True, **kwargs
        )

        # for each of the attributes set it to the problem
        for key, val in out.items():
            if val is not None:
                pop.set(key, val)

        # finally set all the attributes to be evaluated for all individuals
        pop.apply(lambda ind: ind.evaluated.update(out.keys()))
