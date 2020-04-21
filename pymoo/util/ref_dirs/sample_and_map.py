import numpy as np

from pymoo.interface import sample
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.util.reference_direction import ReferenceDirectionFactory, map_onto_unit_simplex


class RandomSamplingAndMap(ReferenceDirectionFactory):

    def __init__(self,
                 n_dim,
                 n_points,
                 **kwargs):
        super().__init__(n_dim, **kwargs)
        self.n_points = n_points

    def _do(self):
        x = sample(LatinHypercubeSampling(), self.n_points - self.n_dim, self.n_dim)
        x = map_onto_unit_simplex(x, "kraemer")
        x = np.row_stack([x, np.eye(self.n_dim)])
        return x
