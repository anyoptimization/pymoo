import numpy as np
from scipy.spatial.distance import cdist

from pymoo.decision_making.decision_making import DecisionMaking, normalize


class CompromiseProgramming(DecisionMaking):

    def __init__(self, metric="euclidean", **kwargs) -> None:
        super().__init__(**kwargs)
        self.metric = metric

    def _do(self, F, **kwargs):

        F, _, ideal_point, _ = normalize(F, self.ideal_point, self.nadir_point, estimate_bounds_if_none=True,
                                         return_bounds=True)

        D = cdist(ideal_point[None, :], F)

        return np.argmin(D)
