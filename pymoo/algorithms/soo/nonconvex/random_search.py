from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.single import SingleObjectiveOutput


class RandomSearch(Algorithm):

    def __init__(self,
                 n_points_per_iteration=100,
                 sampling=FloatRandomSampling(),
                 output=SingleObjectiveOutput(),
                 **kwargs):
        super().__init__(output=output, **kwargs)
        self.n_points_per_iteration = n_points_per_iteration
        self.sampling = sampling

    def _initialize_infill(self):
        return self._infill()

    def _infill(self):
        return self.sampling.do(self.problem, self.n_points_per_iteration)

    def _advance(self, infills=None, **kwargs):
        self.pop = infills if self.opt is None else Population.merge(infills, self.opt)
