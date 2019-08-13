import numpy as np

from pymoo.model.decision_making import DecisionMaking, normalize


class PseudoWeights(DecisionMaking):

    def __init__(self, weights, **kwargs) -> None:
        super().__init__(**kwargs)
        self.weights = weights

    def _do(self, F, return_pseudo_weights=False, **kwargs):

        # here the normalized values are ignored but the ideal and nadir points are estimated to calculate trade-off
        _, norm, ideal_point, nadir_point = normalize(F, self.ideal_point, self.nadir_point,
                                                      estimate_bounds_if_none=True, return_bounds=True)

        # normalized distance to the worst solution
        pseudo_weights = ((nadir_point - F) / norm)

        # normalize weights to sum up to one
        pseudo_weights = pseudo_weights / np.sum(pseudo_weights, axis=1)[:, None]

        # search for the closest individual having this pseudo weights
        I = np.argmin(np.sum(np.abs(pseudo_weights - self.weights), axis=1))

        if return_pseudo_weights:
            return I, pseudo_weights
        else:
            return I



