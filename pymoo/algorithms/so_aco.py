from abc import abstractmethod

import numpy as np

from pymoo.algorithms.so_genetic_algorithm import FitnessSurvival
from pymoo.docs import parse_doc_string
from pymoo.model.algorithm import Algorithm
from pymoo.model.evaluator import set_cv, set_feasibility, Evaluator
from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.optimize import minimize
from pymoo.problems.single.traveling_salesman import create_random_tsp_problem
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.roulette import RouletteWheelSelection


# =========================================================================================================
# Display
# =========================================================================================================


class ACODisplay(SingleObjectiveDisplay):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)


# =========================================================================================================
# Entry
# =========================================================================================================


class Entry:

    def __init__(self, key=None, value=None, heuristic=None) -> None:
        super().__init__()
        self.key = key
        self.value = value
        self.heuristic = heuristic


# =========================================================================================================
# Ant
# =========================================================================================================

class Ant(Individual):

    def __init__(self,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = []
        self.data = {}

        self.problem = None
        self.pheromones = None
        self.alpha = None
        self.beta = None

    def initialize(self, problem, pheromones, alpha, beta):
        self.problem = problem
        self.pheromones = pheromones
        self.alpha = alpha
        self.beta = beta

    def next(self):
        candidates = self._next()

        # information coming from the pheromones
        tau = []
        for entry in candidates:
            key = entry.key
            if not self.pheromones.has(key):
                self.pheromones.initialize(key)
            _tau = self.pheromones.get(key)
            tau.append(_tau)
        tau = np.array(tau)

        # information coming from the heuristics
        eta = np.array([e.heuristic for e in candidates])

        p = tau ** self.alpha + eta ** self.beta
        p = p / p.sum()

        k = RouletteWheelSelection(p).next()

        entry = candidates[k]
        self.notify(entry)

    def has_next(self):
        return self._has_next()

    def notify(self, entry):
        self.path.append(entry)
        self._notify(entry)

    def get_values(self, key="value", as_numpy_array=True, skip_if_none=True):
        ret = [e.__dict__[key] for e in self.path]
        if skip_if_none:
            ret = [e for e in ret if e is not None]
        if as_numpy_array:
            ret = np.array(ret)
        return ret

    def finalize(self):
        self.X = self.get_values()
        self._finalize()

    @abstractmethod
    def _next(self):
        raise Exception("To be implemented")

    @abstractmethod
    def _has_next(self):
        raise Exception("To be implemented")

    def _finalize(self):
        pass

    def _notify(self, entry):
        pass


# =========================================================================================================
# Pheromones
# =========================================================================================================


class Pheromones:

    def __init__(self,
                 ) -> None:
        super().__init__()
        self.data = {}

    def initialize(self, key):
        self.set(key, 1.0)

    def get(self, key):
        return self.data.get(key)

    def has(self, key):
        return key in self.data

    def set(self, key, value):
        self.data[key] = value

    def evaporate(self, rho):
        for key in list(self.data.keys()):
            self.data[key] *= (1 - rho)

    def update(self, keys, value):
        for key in keys:
            self.data[key] += value


# =========================================================================================================
# Mock Evaluator
# =========================================================================================================


class MockEvaluator(Evaluator):

    def __init__(self, **kwargs):
        super().__init__(skip_already_evaluated=False, **kwargs)

    def _eval(self, problem, pop, **kwargs):
        pass


# =========================================================================================================
# Algorithm
# =========================================================================================================


class ACO(Algorithm):

    def __init__(self,
                 ant,
                 pheromones=Pheromones(),
                 n_ants=1,
                 rho=0.1,
                 alpha=1.0,
                 beta=10.0,
                 q=2.0,
                 evaluate_each_ant=True,
                 display=ACODisplay(),
                 **kwargs):
        """

        Parameters
        ----------
        n_ants : int
            Number of ants to be used each iteration

        alpha : float
            Relative importance of pheromone

        beta : float
            Relative importance of heuristic information

        rho : float
            Pheromone residual coefficient

        q : float
            Pheromone intensity

        """

        super().__init__(display=display, **kwargs)

        self.ant = ant
        self.n_ants = n_ants
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.evaluate_each_ant = evaluate_each_ant
        self.pheromones = pheromones

    def _initialize(self):
        self.evaluator = Evaluator() if self.evaluate_each_ant else MockEvaluator()
        self.opt = Population()
        self._next()

    def _next(self):

        ants = []

        # for each ant to be send in an iteration
        for k in range(self.n_ants):

            # create a new ant and send it through the search space
            ant = self.ant()
            ant.initialize(self.problem, self.pheromones, self.alpha, self.beta)
            while ant.has_next():
                ant.next()
            ant.finalize()

            # add the ant to collect all of them
            ants.append(ant)

        colony = Population.create(*ants)
        self.evaluator.eval(self.problem, colony)
        set_cv(colony)
        set_feasibility(colony)

        # do the evaporation after this iteration
        self.pheromones.evaporate(self.rho)

        # now spread the pheromones for each ant depending on performance
        for ant in colony:
            # print(ant.X)
            keys = ant.get_values(key="key", as_numpy_array=False)
            print([round(self.pheromones.get(k), 3) for k in keys])
            value = self.q / ant.F[0]
            self.pheromones.update(keys, value)

        opt = FitnessSurvival().do(problem, Population.merge(colony, self.opt), 1)

        self.pop, self.off = colony, colony
        self.opt = opt

    def _set_optimum(self, **kwargs):
        pass


parse_doc_string(ACO.__init__)


# =========================================================================================================
# TSP Example
# =========================================================================================================


class GraphAnt(Ant):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.not_visited = None

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

        self.not_visited = np.full(self.problem.n_var, True)
        self.notify(Entry(key=None, value=0))

    def _notify(self, entry):
        self.not_visited[entry.value] = False

    def _next(self):
        ret = []

        _current = self.path[-1].value
        remaining = np.where(self.not_visited)[0]

        for city in remaining:
            _next = city
            _heur = 1 / self.problem.D[_current, _next]
            entry = Entry(key=(_current, _next), value=_next, heuristic=_heur)
            ret.append(entry)

        return ret

    def _has_next(self):
        return len(self.path) < self.problem.n_var


if __name__ == "__main__":
    problem = create_random_tsp_problem(10, 100, seed=1)

    algorithm = ACO(GraphAnt, n_ants=5, alpha=1.0, beta=4.0, q=4.0)

    minimize(problem,
             algorithm,
             seed=1,
             verbose=True)
