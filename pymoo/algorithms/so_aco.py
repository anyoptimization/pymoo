from abc import abstractmethod
from copy import copy, deepcopy

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

        F = algorithm.off.get("F")
        self.output.append("it-best", F.min(), width=12)
        self.output.append("it-avg", F.mean(), width=12)


# =========================================================================================================
# Entry
# =========================================================================================================


class Entry:

    def __init__(self, key=None, heuristic=None, pheromone=None) -> None:
        super().__init__()
        self.key = key
        self.heuristic = heuristic
        self.pheromone = pheromone


# =========================================================================================================
# Ant
# =========================================================================================================

class Ant(Individual):

    def __init__(self,
                 alpha=0.1,
                 beta=2.0,
                 q0=None,
                 **kwargs) -> None:
        """


        Parameters
        ----------

        alpha : float
            Relative importance of pheromones

        beta : float
            Relative importance of heuristic information

        q0 : float
            Probability of choosing the maximum values according to selection metric and not doing
            the roulette wheel selection. For Ant Colony System 0.9 was proposed.

        """
        super().__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta
        self.q0 = q0

        self.path = []
        self.data = {}

        self.problem = None
        self.pheromones = None

    def initialize(self, problem, pheromones):
        self.problem = problem
        self.pheromones = pheromones

    def next(self):
        candidates = self._next()

        # information coming from the pheromones
        tau = []
        for entry in candidates:
            key = entry.key

            # the first time this key is accessed it is initialized before - if sparse implementation is used
            if not self.pheromones.has(key):
                self.pheromones.initialize(key)

            # get the pheromones on the field and add it to the list
            _tau = self.pheromones.get(key)
            tau.append(_tau)

        tau = np.array(tau)

        # information coming from the heuristics
        eta = np.array([e.heuristic for e in candidates])

        # p = tau ** self.alpha + eta ** self.beta
        p = tau + eta ** self.beta
        p = p / p.sum()

        if self.q0 is not None and np.random.random() <= self.q0:
            k = p.argmax()
        else:
            k = RouletteWheelSelection(p).next()

        entry = candidates[k]
        self.notify(entry)

    def has_next(self):
        return self._has_next()

    def last(self):
        if len(self.path) > 0:
            return self.path[-1]
        else:
            return None

    def notify(self, entry):
        self.path.append(entry)
        self._notify(entry)

    def get_values(self, key="key", as_numpy_array=True):
        ret = [e.__dict__[key] for e in self.path]
        if as_numpy_array:
            ret = np.array(ret)
        return ret

    def finalize(self):
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


def get_symmetric(key):
    if isinstance(key, tuple) or isinstance(key, tuple):
        if len(key) == 2:
            return min(key), max(key)

    raise Exception("Symmetric pheromones can only be used for a two-dimensional matrix.")


class Pheromones:

    def __init__(self,
                 tau_init=0.1,
                 rho=0.1,
                 symmetric=True
                 ) -> None:
        """

        Parameters
        ----------
        rho : float
            The evaporation parameter.

        symmetric : bool
            Whether the pheromones should be set symmetrically

        """
        super().__init__()
        self.tau_init = tau_init
        self.tau_init_evaporated = tau_init
        self.rho = rho
        self.symmetric = symmetric

        self.data = {}

    def initialize(self, key):
        self.set(key, self.tau_init_evaporated)

    def _get(self, key):
        key = get_symmetric(key) if self.symmetric else key
        return key

    def _set(self, key, value):
        key = get_symmetric(key) if self.symmetric else key
        self.data[key] = value

    def get(self, key):
        return self.data.get(self._get(key))

    def has(self, key):
        return self._get(key) in self.data

    def set(self, key, value):
        self._set(key, value)

    def evaporate(self):
        keys = list(self.data.keys())

        for key in keys:
            value = self.data[self._get(key)]
            self._set(key, value * (1 - self.rho))

        self.tau_init_evaporated = self.tau_init_evaporated * (1 - self.rho)

    def update(self, key, value):
        self.data[self._get(key)] += value


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
                 pheromones,
                 n_ants=10,
                 global_update='best',
                 local_update=True,
                 evaluate_each_ant=True,
                 display=ACODisplay(),
                 **kwargs):
        """

        Parameters
        ----------

        ant : class
            An objective defining the behavior of each ant

        pheromones : class
            The pheromone implementation storing the amount of pheromones on each edge,
            the evaporation and update.

        n_ants : int
            Number of ants to be used each iteration

        global_update : {'all', 'it-best', 'best'}

        """

        super().__init__(display=display, **kwargs)

        # make the ant always to be a function being able to call
        if not callable(ant):
            proto = deepcopy(ant)

            def new_ant():
                return deepcopy(proto)

            ant = new_ant

        self.ant = ant
        self.pheromones = pheromones
        self.n_ants = n_ants
        self.global_update = global_update
        self.local_update = local_update
        self.evaluate_each_ant = evaluate_each_ant

    def _initialize(self):
        self.evaluator = Evaluator(skip_already_evaluated=False) if self.evaluate_each_ant else MockEvaluator()
        self.opt = Population()
        self._next()

    def _next(self):

        # initialize all ants to be used in this iteration
        ants = []
        for k in range(self.n_ants):
            ant = self.ant()
            ant.initialize(self.problem, self.pheromones)
            ants.append(ant)

        active = list(range(self.n_ants))

        while len(active) > 0:

            for k in active:
                ant = ants[k]

                if ant.has_next():
                    ant.next()

                    if self.local_update:
                        e = ant.last()
                        if e is None or e.pheromone is None:
                            raise Exception("For a local update the ant has to set the pheromones when notified.")
                        else:
                            self.pheromones.set(e.key, self.pheromones.get(e.key) * ant.alpha + e.pheromone * ant.alpha)
                            # self.pheromones.update(e.key, e.pheromone * ant.alpha)

                else:
                    ant.finalize()
                    active = [i for i in active if i != k]

        colony = Population.create(*ants)

        # this evaluation can be disabled or faked if evaluate_each_ant is false - then the finalize method of the
        # ant has to set the objective and/or constraint values accordingly
        self.evaluator.eval(self.problem, colony)
        set_cv(colony)
        set_feasibility(colony)

        # set the current best including the new colony
        opt = FitnessSurvival().do(problem, Population.merge(colony, self.opt), 1)

        # do the evaporation after this iteration
        self.pheromones.evaporate()

        # select the ants to be used for the global pheromone update
        if self.global_update == "all":
            ants_to_update = colony
        elif self.global_update == "it-best":
            ants_to_update = FitnessSurvival().do(problem, colony, 1)
        elif self.global_update == "best":
            ants_to_update = self.opt
        else:
            raise Exception("Unknown value for global updating the pheromones!")

        # now spread the pheromones for each ant depending on performance
        for ant in ants_to_update:
            for e in ant.path:
                if e.pheromone is None:
                    raise Exception("The ant has to set the pheromone of each entry in the path.")
                else:
                    self.pheromones.update(e.key, e.pheromone * pheromones.rho)

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
        self.costs = 0.0

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.not_visited = np.full(self.problem.n_var, True)

    def _notify(self, entry):
        self.not_visited[list(entry.key)] = False

        a, b = entry.key
        self.costs += self.problem.D[a, b]

    def heuristic(self, key):
        _current, _next = key
        _heur = 1 / self.problem.D[_current, _next]
        return _heur

    def _next(self):
        ret = []

        if len(self.path) == 0:
            _current = np.random.randint(self.problem.n_var)
            self.not_visited[_current] = False
        else:
            _, _current = self.path[-1].key

        remaining = np.where(self.not_visited)[0]

        if len(remaining) == 0:
            return_to_first, _ = self.path[0].key
            remaining = [return_to_first]

        for e in remaining:
            _next = int(e)
            _key = (_current, _next)

            # the heuristic value for this edge - default 1 / cost(i,j)
            _heur = self.heuristic(_key)

            # if local pheromone updates are enabled this needs to be set here
            _pheromone = self.pheromones.tau_init

            entry = Entry(key=_key, heuristic=_heur, pheromone=_pheromone)
            ret.append(entry)

        return ret

    def _finalize(self):

        self.X = np.array([e.key[0] for e in self.path])
        assert (len(self.X) == self.problem.n_var)

        self.F = np.array([self.costs])

        for entry in self.path:
            entry.pheromone = 1 / self.costs

    def _has_next(self):
        return len(self.path) < self.problem.n_var


def solve_nearest_neighbor(tsp, return_time=False):
    n_cities = tsp.n_var
    time = 0.0

    # initialize the array to store what cities have been visited
    visited = np.full(n_cities, False)

    # randomly choose the first city to start
    _start = np.random.randint(n_cities)
    visited[_start] = True

    # create the initial tour
    tour = [_start]

    # until we have not visited all cities
    while len(tour) < n_cities:
        _current = tour[-1]

        remaining = np.where(~visited)[0]
        d = tsp.D[_current][remaining]

        k = d.argmin()
        closest = remaining[k]
        visited[closest] = True
        time += d[k]

        tour.append(closest)

    first, last = tour[0], tour[-1]
    time += tsp.D[last][first]

    tour = np.array(tour)

    if not return_time:
        return tour
    else:
        return tour, time


if __name__ == "__main__":
    problem = create_random_tsp_problem(50, 100, seed=1)

    import numpy as np
    from pymoo.model.repair import Repair


    class StartFromZeroRepair(Repair):

        def _do(self, problem, pop, **kwargs):
            X = pop.get("X")
            I = np.where(X == 0)[1]

            for k in range(len(X)):
                i = I[k]
                x = X[k]
                _x = np.concatenate([x[i:], x[:i]])
                pop[k].set("X", _x)

            return pop


    from pymoo.algorithms.so_genetic_algorithm import GA
    from pymoo.optimize import minimize
    from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
    from pymoo.problems.single.traveling_salesman import create_random_tsp_problem
    from pymoo.util.termination.default import SingleObjectiveDefaultTermination

    algorithm = GA(
        pop_size=20,
        sampling=get_sampling("perm_random"),
        crossover=get_crossover("perm_erx"),
        mutation=get_mutation("perm_inv"),
        repair=StartFromZeroRepair(),
        eliminate_duplicates=True
    )

    # if the algorithm did not improve the last 200 generations then it will terminate (and disable the max generations)
    termination = SingleObjectiveDefaultTermination(n_last=200, n_max_gen=np.inf)

    # res = minimize(
    #     problem,
    #     algorithm,
    #     termination,
    #     seed=1,
    # )
    #
    # print("Traveling Time:", np.round(res.F[0], 3))
    # print("Function Evaluations:", res.algorithm.evaluator.n_eval)

    ant = lambda: GraphAnt(alpha=0.1, beta=2.0, q0=0.9)

    nn_times = []
    for k in range(10):
        _, time = solve_nearest_neighbor(problem, return_time=True)
        nn_times.append(time)
    time = min(nn_times)

    tau_init = (problem.n_var * time) ** -1

    pheromones = Pheromones(tau_init=tau_init, rho=0.1)

    for i in range(problem.n_var):
        for j in range(i + 1, problem.n_var):
            pheromones.initialize((i, j))

    algorithm = ACO(ant, pheromones, n_ants=10, global_update="best", local_update=True)

    minimize(problem,
             algorithm,
             ("n_gen", 300),
             seed=1,
             verbose=True)
