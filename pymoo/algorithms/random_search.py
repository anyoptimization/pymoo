from pymoo.core.algorithm import Algorithm
from pymoo.operators.sampling import Sampling


class RandomSearch(Algorithm):

    def __init__(self,
                 sampling: Sampling = Sampling(),
                 n_samples: int = 100,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.sampling = sampling
        self.n_samples = n_samples

    def advance(self):
        sols = self.sampling.sample(self.problem, self.n_samples)
        sols = yield from self.evaluator.send(sols)
        return sols
