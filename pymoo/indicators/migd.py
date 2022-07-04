import numpy as np

from pymoo.core.callback import Callback
from pymoo.indicators.igd import IGD


class MIGD(Callback):

    def __init__(self, reevaluate=True) -> None:
        """
        Mean Inverted Generational Distance (MIGD)

        For dynamic optimization problems the performance metric needs to involve the IGD value over time as the
        problem is changing. Thus, the performance needs to be evaluated in each iteration for which
        defining a callback is ideal.

        """

        super().__init__()

        # whether the MIGD should be based on reevaluated solutions
        self.reevaluate = reevaluate

        # the list where each of the recordings are stored: timesteps and igd
        self.records = []

    def update(self, algorithm, **kwargs):

        # the problem to be solved
        problem = algorithm.problem
        assert problem.n_constr == 0, "The current implementation only works for unconstrained problems!"

        # the current time
        t = problem.time

        # the current pareto-front of the problem (at the specific time step)
        pf = problem.pareto_front()

        # the current population of the algorithm
        pop = algorithm.pop

        # if the callback should reevaluate to match the current time step and avoid deprecated values
        if self.reevaluate:
            X = pop.get("X")
            F = problem.evaluate(X, return_values_of=["F"])
        else:
            F = pop.get("F")

        # calculate the current igd values
        igd = IGD(pf).do(F)

        self.records.append((t, igd))

    def value(self):
        return np.array([igd for _, igd in self.records]).mean()
