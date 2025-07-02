import numpy as np

from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.util.reference_direction import ReferenceDirectionFactory, map_onto_unit_simplex


class RandomSamplingAndMap(ReferenceDirectionFactory):

    def __init__(self,
                 n_dim,
                 n_points,
                 **kwargs):
        super().__init__(n_dim, **kwargs)
        self.n_points = n_points

    def _do(self, random_state=None):
        problem = Problem(n_var=self.n_dim, xl=0.0, xu=1.0)
        sampling = LatinHypercubeSampling()

        x = sampling(problem, self.n_points - self.n_dim, to_numpy=True, random_state=random_state)
        x = map_onto_unit_simplex(x, "kraemer")
        x = np.vstack([x, np.eye(self.n_dim)])
        return x
