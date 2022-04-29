import numpy as np

from pymoo.core.algorithm import Algorithm
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.replacement import is_better
from pymoo.termination.default import DefaultSingleObjectiveTermination


class ExponentialSearch(Algorithm):

    def __init__(self, delta=0.05, **kwargs):
        super().__init__(**kwargs)
        self.termination = DefaultSingleObjectiveTermination()
        self.alpha = delta
        self.point = None

    def _setup(self, problem, x0=None, **kwargs):
        msg = "Only problems with one variable, one objective and no constraints can be solved!"
        assert problem.n_var == 1 and not problem.has_constraints() and problem.n_obj == 1, msg
        self.point = x0

    def _initialize_infill(self):
        self.step_size = self.alpha

        if self.point is None:
            return Population.new(X=np.copy(self.problem.xl[None, :]))
        else:
            return Population.create(self.point)

    def step(self):
        alpha, max_alpha = self.alpha, self.problem.xu[0]

        if alpha > max_alpha:
            alpha = max_alpha

        infill = Individual(X=np.array([alpha]))
        self.evaluator.eval(self.problem, infill)
        self.pop = Population.merge(self.pop, infill)[-10:]

        if is_better(self.point, infill, eps=0.0) or alpha == max_alpha:
            self.termination.force_termination = True
            return

        self.point = infill
        self.alpha *= 2
