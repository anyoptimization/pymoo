import numpy as np

from pymoo.algorithms.so_local_search import LocalSearch
from pymoo.docs import parse_doc_string
from pymoo.model.individual import Individual
from pymoo.model.population import Population, pop_from_array_or_individual
from pymoo.model.replacement import is_better
from pymoo.model.termination import Termination
from pymoo.operators.repair.out_of_bounds_repair import repair_out_of_bounds_manually
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.termination.default import SingleObjectiveDefaultTermination


# =========================================================================================================
# Implementation
# =========================================================================================================


class PatternSearchDisplay(SingleObjectiveDisplay):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("T", algorithm.T)


class PatternSearchTermination(Termination):

    def __init__(self, min_T=1e-5, **kwargs):
        super().__init__()
        self.default = SingleObjectiveDefaultTermination(**kwargs)
        self.min_T = min_T

    def do_continue(self, algorithm):
        decision_default = self.default.do_continue(algorithm)
        if algorithm.T < self.min_T:
            return decision_default
        else:
            return True


class PatternSearch(LocalSearch):

    def __init__(self,
                 T=0.1,
                 a=2,
                 display=PatternSearchDisplay(),
                 **kwargs):
        super().__init__(display=display, **kwargs)
        self.T = T
        self.a = a
        self.default_termination = PatternSearchTermination(x_tol=1e-6, f_tol=1e-6, nth_gen=1, n_last=2)

    def _next(self):
        # the current best solution found so far
        best = self.opt[0]

        # first do the exploration move
        opt = self._exploration_move(best)

        # if the exploration move could not improve the current solution
        if opt == best:
            self.T = self.T / 2

        # if the move brought up a new solution -> perform a line search
        else:
            self._pattern_move(best, opt)

    def _pattern_move(self, old_best, new_best):
        _current, _next = old_best, new_best

        while True:
            X = _current.X + self.a * (_next.X - _current.X)
            xl, xu = self.problem.bounds()
            X = repair_out_of_bounds_manually(X, xl, xu)
            tentative = Individual(X=X)

            self.evaluator.eval(self.problem, tentative, algorithm=self)
            self.pop = Population.merge(self.pop, tentative)

            # if the tentative could not further improve the old best
            if not is_better(tentative, _next):
                break
            else:
                # if we have improved
                _current, _next = _next, tentative

    def _exploration_move(self, opt):
        xl, xu = self.problem.bounds()

        def step(x, sign):
            # copy to not modify the original value
            X = np.copy(x)

            # add the value in the normalized space to the k-th component
            X[k] = X[k] + (sign * self.T) * (xu[k] - xl[k])

            # repair if out of bounds
            X = repair_out_of_bounds_manually(X, xl, xu)

            # return the new solution as individual
            mutant = pop_from_array_or_individual(X)[0]

            return mutant

        for k in range(self.problem.n_var):

            # randomly assign + or - as a sign
            sign = 1 if np.random.random() < 0.5 else -1

            # create the the individual and evaluate it
            mutant = step(opt.X, sign)
            self.evaluator.eval(self.problem, mutant, algorithm=self)
            self.pop = Population.merge(self.pop, mutant)

            if is_better(mutant, opt):
                opt = mutant

            else:

                # now try the other sign if there was no improvement
                mutant = step(opt.X, -1 * sign)
                self.evaluator.eval(self.problem, mutant, algorithm=self)
                self.pop = Population.merge(self.pop, mutant)

                if is_better(mutant, opt):
                    opt = mutant

        return opt


parse_doc_string(PatternSearch.__init__)
