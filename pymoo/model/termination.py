import numpy as np

from pymoo.performance_indicator.igd import IGD
from pymoo.performance_indicator.igd_plus import IGDPlus
from pymoo.util.normalization import normalize


class Termination:

    def __init__(self) -> None:
        super().__init__()
        self.force_termination = False

    def do_continue(self, D):
        return (not self.force_termination) and self._do_continue(D)

    def has_finished(self, D):
        return not self.do_continue(D)


class MaximumFunctionCallTermination(Termination):

    def __init__(self, n_max_evals) -> None:
        super().__init__()
        self.n_max_evals = n_max_evals

    def _do_continue(self, algorithm):
        return algorithm.evaluator.n_eval < self.n_max_evals


class MaximumGenerationTermination(Termination):

    def __init__(self, n_max_gen) -> None:
        super().__init__()
        self.n_max_gen = n_max_gen

    def _do_continue(self, algorithm):
        return algorithm.n_gen < self.n_max_gen


class ToleranceBasedTermination(Termination):

    def __init__(self, tol=0.001, n_last=20, n_max_gen=1000, nth_gen=5) -> None:
        super().__init__()
        self.tol = tol
        self.nth_gen = nth_gen
        self.n_last = n_last
        self.n_max_gen = n_max_gen
        self.history, self.n_gen = [], None

    def _store(self, pop):
        return pop

    def _decide(self):
        return True

    def _do_continue(self, algorithm):

        # the current generation of the algorithm
        self.n_gen = algorithm.n_gen

        # if the fallback stop in generation is enabled check it
        if self.n_max_gen is not None and self.n_gen >= self.n_max_gen:
            return False

        # store the current data in the history
        self.history.insert(0, self._store(algorithm))

        # truncate everything after window
        self.history = self.history[:self.n_last]

        # at least two sets need to be compared
        if len(self.history) >= self.n_last and self.n_gen % self.nth_gen == 0:
            return self._decide()
        else:
            return True


class DesignSpaceToleranceTermination(ToleranceBasedTermination):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.xl = None
        self.xu = None

    def _store(self, algorithm):

        if self.xl is None:
            self.xl = algorithm.problem.xl

        if self.xu is None:
            self.xu = algorithm.problem.xu

        return algorithm.pop.get("X")

    def _decide(self):
        H = [normalize(e, x_min=self.xl, x_max=self.xu) for e in self.history]

        perf = np.full(self.n_last - 1, np.inf)
        for k in range(self.n_last - 1):
            current, last = H[k], H[k + 1]
            perf[k] = IGD(current).calc(last)

        return perf.std() > self.tol


class ObjectiveSpaceToleranceTermination(ToleranceBasedTermination):

    def _store(self, algorithm):
        return algorithm.pop.get("F"), algorithm.pop.get("CV")

    def _decide(self):

        # get the data of the latest generation
        c_F, c_CV = self.history[0]

        # extract the constraint violation information
        CV = np.array([e[1].min() for e in self.history])

        # if some constraints were violated in the window
        if CV.max() > 0:

            # however if in the current generation a solution is feasible - continue
            if c_CV.min() == 0:
                return True

            # otherwise still no feasible solution was found, apply the CV tolerance
            else:
                # normalize by the maximum minimum CV in each window
                CV = CV / CV.max()
                CV = np.array([CV[k + 1] - CV[k] for k in range(self.n_last - 1)])
                return CV.max() > self.tol

        else:

            F = [normalize(e[0], c_F.min(axis=0), c_F.max(axis=0)) for e in self.history]

            # the metrics to keep track of
            perf = np.full(self.n_last - 1, np.inf)
            ideal, nadir = perf.copy(), perf.copy()

            for k in range(self.n_last - 1):
                current, last = F[k], F[k + 1]

                ideal[k] = (current.min(axis=0) - last.min(axis=0)).max()
                nadir[k] = (current.max(axis=0) - last.max(axis=0)).max()
                perf[k] = IGDPlus(current).calc(last)

            return ideal.max() > self.tol or nadir.max() > self.tol or perf.mean() > self.tol


class IGDTermination(Termination):

    def __init__(self, min_igd, pf) -> None:
        super().__init__()
        if pf is None:
            raise Exception("You can only use IGD termination criteria if the pareto front is known!")

        self.obj = IGD(pf)
        self.igd = min_igd

    def _do_continue(self, algorithm):
        F = algorithm.pop.get("F")
        return self.obj.calc(F) > self.igd
