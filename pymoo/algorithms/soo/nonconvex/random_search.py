from pymoo.model.algorithm import Algorithm
from pymoo.model.population import Population
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.util.display import SingleObjectiveDisplay


class RandomSearch(Algorithm):

    def __init__(self,
                 n_points_per_iteration=100,
                 sampling=LatinHypercubeSampling(),
                 display=SingleObjectiveDisplay(),
                 **kwargs):
        super().__init__(display=display, **kwargs)
        self.n_points_per_iteration = n_points_per_iteration
        self.sampling = sampling

    def _initialize_infill(self):
        return self._infill()

    def _infill(self):
        return self.sampling.do(self.problem, self.n_points_per_iteration)

    def _advance(self, infills=None):
        self.pop = infills if self.opt is None else Population.merge(infills, self.opt)
