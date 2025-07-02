import numpy as np

from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.population import Population, merge
from pymoo.util import default_random_state
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class Truncation:

    def __call__(self, sols, k):
        pass


class RandomTruncation(Truncation):

    @default_random_state
    def __call__(self, sols, k, random_state=None):
        return random_state.choice(sols, size=k, replace=False)


class SurvivalTruncation(Truncation):

    def __init__(self, survival, problem=None) -> None:
        super().__init__()
        self.survival = survival

        if problem is None:
            from pymoo.core.problem import Problem
            problem = Problem()

        self.problem = problem

    def __call__(self, sols, k):
        return self.survival.do(self.problem, sols, n_survive=k)


class Archive(Population):

    def __new__(cls,
                individuals=[],
                max_size=None,
                truncate_size=None,
                truncation=RandomTruncation(),
                duplicate_elimination=DefaultDuplicateElimination(epsilon=1e-32)):

        obj = super().__new__(cls, individuals)
        obj.max_size = max_size
        obj.truncate_size = min(max_size, truncate_size) if truncate_size is not None else max_size
        obj.truncation = truncation
        obj.duplicate_elimination = duplicate_elimination

        return obj

    def __array_finalize__(self, obj):
        if obj is None:  # __new__ handles instantiation
            return

        max_size = getattr(obj, 'max_size', None)
        truncate_size = getattr(obj, 'truncate_size', None)
        truncation = getattr(obj, 'truncation', RandomTruncation())
        duplicate_elimination = getattr(obj, 'duplicate_elimination', DefaultDuplicateElimination(epsilon=1e-32))

        self.max_size = max_size
        self.truncate_size = min(max_size, truncate_size) if truncate_size is not None else max_size
        self.truncation = truncation
        self.duplicate_elimination = duplicate_elimination

    def add(self, sols):

        if len(self) > 0:
            sols = merge(self, sols)

        opt = self._find_opt(sols)

        if self.duplicate_elimination:
            opt = self.duplicate_elimination.do(opt)

        if self.max_size and len(opt) > self.max_size:
            opt = self.truncation(opt, self.truncate_size)

        cls = self.__class__
        obj = cls.__new__(cls, individuals=opt, **self.view(Archive).__dict__)
        return obj

    def _find_opt(self, sols):
        pass


class VoidArchive(Archive):

    def add(self, sols):
        return self


class SingleObjectiveArchive(Archive):

    def __new__(cls, max_size=10, **kwargs):
        return super().__new__(cls, max_size=max_size, **kwargs).view(cls)

    def _find_opt(self, sols):
        feas = sols.get("feas")

        if np.any(feas):
            sols = sols[feas]

            f = sols.get("f")
            I, = np.where(f == f[f.argmin()])

        else:
            cv = sols.get("cv")
            I, = np.where(cv == cv[cv.argmin()])

        return sols[I]


class MultiObjectiveArchive(Archive):

    def __new__(cls, max_size=200, truncate_size=100, **kwargs):
        return super().__new__(cls,
                               max_size=max_size,
                               truncate_size=truncate_size,
                               **kwargs)

    def _find_opt(self, sols):
        feas = sols.get("feas")

        if np.any(feas):
            sols = sols[feas]

            F = sols.get("F")
            I = NonDominatedSorting().do(F, only_non_dominated_front=True)
        else:
            cv = sols.get("cv")
            I, = np.where(cv == cv[cv.argmin()])

        return sols[I]


def default_archive(problem, **kwargs):
    if problem.n_obj == 1:
        return SingleObjectiveArchive(**kwargs)

    elif problem.n_obj == 2:
        from pymoo.algorithms.moo.sms import LeastHypervolumeContributionSurvival
        survival = LeastHypervolumeContributionSurvival()
        return MultiObjectiveArchive(truncation=SurvivalTruncation(survival, problem=problem), **kwargs)

    elif problem.n_obj >= 3:
        from pymoo.algorithms.moo.spea2 import SPEA2Survival
        survival = SPEA2Survival()
        return MultiObjectiveArchive(truncation=SurvivalTruncation(survival, problem=problem), **kwargs)
