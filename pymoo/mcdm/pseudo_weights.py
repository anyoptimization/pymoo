import numpy as np

from pymoo.core.decision_making import DecisionMaking


class PseudoWeights(DecisionMaking):

    def __init__(self, weights, **kwargs) -> None:
        super().__init__(**kwargs)
        self.weights = weights

    def _do(self, F, return_pseudo_weights=False, **kwargs):
        ideal, nadir = self.ideal, self.nadir

        if ideal is None:
            ideal = F.min(axis=0)
        if nadir is None:
            nadir = F.max(axis=0)

        # normalized distance to the worst solution
        pseudo_weights = ((nadir - F) / (nadir - ideal))

        # normalize weights to sum up to one
        pseudo_weights = pseudo_weights / np.sum(pseudo_weights, axis=1)[:, None]

        # search for the closest individual having this pseudo weights
        I = np.argmin(np.sum(np.abs(pseudo_weights - self.weights), axis=1))

        if return_pseudo_weights:
            return I, pseudo_weights
        else:
            return I
