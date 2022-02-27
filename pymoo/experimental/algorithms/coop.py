from copy import deepcopy

import numpy as np

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm
from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.problems.meta import MetaProblem
from pymoo.problems.static import StaticProblem
from pymoo.util.archive import SingleObjectiveArchive
from pymoo.util.display import SingleObjectiveDisplay, MultiObjectiveDisplay
from pymoo.util.termination.default import SingleObjectiveDefaultTermination, MultiObjectiveDefaultTermination
from pymoo.util.termination.no_termination import NoTermination


# ---------------------------------------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------------------------------------


class Coevolution(MetaProblem):

    def __init__(self, problem, groups=None, **kwargs) -> None:
        super().__init__(problem, **kwargs)
        self.groups = groups
        self.k = None

    def active(self):
        return self.k is not None

    def num_of_groups(self):
        return len(self.groups)

    def vars(self):
        if self.k is None:
            return np.full(self.problem.n_var, True)
        else:
            return self.groups[self.k]["vars"]

    def weights(self):
        if self.k is None:
            return np.ones(self.problem.n_var)
        else:
            return self.groups[self.k]["weights"]

    def update(self, k):
        self.k = k

        if k is None:
            self.n_var = self.problem.n_var
            self.xl = self.problem.xl
            self.xu = self.problem.xu
        else:
            vars = self.vars()
            self.n_var = len(vars)
            self.xl = self.problem.xl[vars]
            self.xu = self.problem.xu[vars]

    def reset(self):
        self.update(None)


# ---------------------------------------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------------------------------------


class CoevolutionaryIndividual(Individual):

    def __init__(self, coevo, **kwargs) -> None:
        super().__init__(**kwargs)
        self.coevo = coevo

    @property
    def X(self):

        if self.coevo.active():
            vars = self.coevo.vars()
            return self._X[vars] * self.coevo.weights()
        else:
            return self._X

    def copy(self, other=None, deep=False):
        obj = self.__class__(self.coevo)

        kwargs = self.__dict__ if other is None else other.__dict__
        for k, v in kwargs.items():
            if not deep:
                obj.__dict__[k] = v
            else:
                obj.__dict__[k] = deepcopy(v)
        return obj


# ---------------------------------------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------------------------------------

class CooperativeCoevolution(Algorithm):

    def __init__(self, n_groups=5, algorithm=None, **kwargs):
        super().__init__(**kwargs)

        # the default algorithm that should be used for each group
        if algorithm is None:
            algorithm = DE()
            # algorithm = NSGA2()

        self.algorithm = algorithm

        # the number of groups to be used
        self.n_groups = n_groups

        # this will be the adaptive co-evolutionary problem
        self.coevo = None

        # this stores the algorithm object for the evolutions to be run
        self.evos = None

        # the infill solutions returned by each of the algorithms
        self.infills = None

        # an archive to store the best solutions
        self.archive = None

    def _setup(self, problem, **kwargs):
        if problem.n_obj == 1:
            self.default_termination = SingleObjectiveDefaultTermination()
            self.display = SingleObjectiveDisplay()
            self.archive = SingleObjectiveArchive()
        else:
            self.default_termination = MultiObjectiveDefaultTermination()
            self.display = MultiObjectiveDisplay()
            self.archive = SingleObjectiveArchive()

        super()._setup(problem, **kwargs)

        # create the groups or sub-population for the co-evolution
        groups = self._create_groups(self.n_groups)

        # initialize the co-evolution problem object
        self.coevo = Coevolution(problem, groups)

        # bind the adaptive individual to the current evaluator
        self.evaluator.individual = CoevolutionaryIndividual(self.coevo)

        self.evos = []

        for k in range(self.n_groups):
            algorithm = deepcopy(self.algorithm)
            algorithm.termination = NoTermination()

            self.coevo.update(k)
            algorithm.setup(self.coevo, **kwargs)

            self.evos.append(algorithm)

        self.coevo.reset()

    def _initialize_infill(self):
        return self._infill()

    def _initialize_advance(self, infills=None, **kwargs):
        self._advance(infills=infills, **kwargs)

    def _infill(self):

        infills = []

        n_infills = None

        for k in range(self.n_groups):

            # change the problem and individuals to be of this group
            self.coevo.update(k)

            # create infill solutions from this population
            pop = self.evos[k].infill()

            # pop = pop[np.random.permutation(len(pop))]

            if n_infills is None:
                n_infills = len(pop)
            else:
                assert len(pop) == n_infills, "All algorithms must return the same amount of infill solutions!"

            infills.append(pop)

        X = np.zeros((n_infills, self.problem.n_var))

        for k in range(self.n_groups):
            vars = self.coevo.groups[k]["vars"]
            X[:, vars] = infills[k].get("X")

        self.coevo.reset()

        self.infills = infills

        return Population.new(X=X)

    def _advance(self, infills=None, **kwargs):

        F, G, H = infills.get("F", "G", "H")

        k = np.random.randint(0, self.n_groups)

        # advance each of the algorithm for the specific sub-population
        for k in range(self.n_groups):

            self.coevo.update(k)

            evo_infills = self.infills[k]

            static = StaticProblem(self.coevo, F=F, G=G, H=H)
            Evaluator().eval(static, evo_infills)

            self.evos[k].advance(infills=evo_infills, **kwargs)

        # reset everything to have all variables active during advance (also for duplicate elimination)
        self.coevo.reset()

        # add solutions to the local archive
        self.archive.add(infills)

    def _create_groups(self, n_groups):
        n = self.problem.n_var

        if n_groups > n:
            n_groups = n

        p = np.random.permutation(n)
        vars_per_group = int(n / n_groups)

        groups = []

        for k in range(n_groups):
            vars = p[k * vars_per_group:(k + 1) * vars_per_group]
            weights = np.ones(len(vars))

            groups.append(dict(index=k, vars=vars, weights=weights))

        return groups

    def _set_optimum(self):
        self.opt = self.archive.sols
