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


# =========================================================================================================
# Display
# =========================================================================================================

class ACODisplay(SingleObjectiveDisplay):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)


# =========================================================================================================
# Ant
# =========================================================================================================

class Ant(Individual):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = []
        self.data = {}
        self.problem = None

    def initialize(self, problem):
        self.problem = problem

    def start(self):
        return self._start()

    def next(self):
        if len(self.path) == 0:
            val = self.start()
        else:
            val = self._next()

        self.path.append(val)

    def has_next(self):
        return self._has_next()

    def finalize(self):
        self.X = np.array(self.path)
        self._finalize()

    def _start(self):
        raise Exception("To be implemented")

    def _next(self):
        raise Exception("To be implemented")

    def _has_next(self):
        raise Exception("To be implemented")

    def _finalize(self):
        raise Exception("To be implemented")


# =========================================================================================================
# Algorithm
# =========================================================================================================


class AntEvaluator(Evaluator):

    def __init__(self, **kwargs):
        super().__init__(skip_already_evaluated=False, **kwargs)

    def _eval(self, problem, pop, **kwargs):
        pass


class ACO(Algorithm):

    def __init__(self,
                 ant,
                 n_ants=1,
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

    def _initialize(self):
        self.evaluator = AntEvaluator()
        self.opt = Population()
        self._next()

    def _next(self):

        ants = []

        # for each ant to be send in an iteration
        for k in range(self.n_ants):

            # create a new ant and send it through the search space
            ant = self.ant()
            ant.initialize(self.problem)
            while ant.has_next():
                ant.next()
            ant.finalize()

            # add the ant to collect all of them
            ants.append(ant)

        colony = Population.create(*ants)
        self.evaluator.eval(self.problem, colony)
        set_cv(colony)
        set_feasibility(colony)

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
        self.time = 0
        self.cities = None

    def initialize(self, problem):
        super().initialize(problem)
        self.cities = self.problem.n_var

    def _start(self):
        return np.random.choice(np.arange(self.cities))

    def _next(self):
        not_visited = np.full(self.problem.n_var, True)
        not_visited[self.path] = False

        remaining = np.where(not_visited)[0]

        _next = np.random.choice(remaining)
        _current = self.path[-1]
        self.time += self.problem.D[_current, _next]

        return _next

    def _has_next(self):
        return len(self.path) < self.problem.n_var

    def _finalize(self):
        self.F = np.array([self.time])


if __name__ == "__main__":
    problem = create_random_tsp_problem(50, 100, seed=1)

    aco = ACO(GraphAnt, n_ants=100)

    minimize(problem, aco, seed=1, verbose=True)
