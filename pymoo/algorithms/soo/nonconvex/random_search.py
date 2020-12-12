from pymoo.model.algorithm import Algorithm
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

    def _initialize(self):
        self._next()

    def _next(self):
        pop = self.sampling.do(self.problem, self.n_points_per_iteration)
        self.evaluator.eval(self.problem, pop, algorithm=self)
        self.pop = pop
