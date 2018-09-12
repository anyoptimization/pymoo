import numpy as np

from pymoo.algorithms.nsga3 import NSGA3
from pymoo.operators.survival.aspiration_point_survival import AspirationPointSurvival
from pymoo.util.reference_direction import UniformReferenceDirectionFactory


class RNSGA3(NSGA3):

    def __init__(self,
                 ref_points,
                 pop_per_ref_point,
                 mu=0.1,
                 **kwargs):

        n_obj = ref_points.shape[1]
        n_ref_points = ref_points.shape[0]

        # add the aspiration point lines
        aspiration_ref_dirs = []
        for i in range(n_ref_points):
            ref_dirs = UniformReferenceDirectionFactory(n_dim=n_obj, n_points=pop_per_ref_point).do()
            aspiration_ref_dirs.extend(ref_dirs)
        aspiration_ref_dirs = np.array(aspiration_ref_dirs)

        kwargs['ref_dirs'] = aspiration_ref_dirs
        super().__init__(**kwargs)

        # create the survival strategy
        self.survival = AspirationPointSurvival(ref_points, aspiration_ref_dirs, mu=mu)

    def _solve(self, problem, termination):

        if self.survival.ref_points.shape[1] != problem.n_obj:
            raise Exception("Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                            (self.survival.ref_points.shape[1], problem.n_obj))

        return super()._solve(problem, termination)