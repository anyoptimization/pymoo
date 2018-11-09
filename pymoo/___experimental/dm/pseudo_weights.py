import numpy as np

from pymoo.dm.decision_making import DecisionMaking


class PseudoWeights(DecisionMaking):

    def __init__(self, weights) -> None:
        super().__init__()
        self.weights = weights
        self.pseudo_weights = None

    def do(self, F, **kwargs):
        # get minimum and maximum for each objective
        F_min = np.min(F, axis=0)
        F_max = np.max(F, axis=0)

        # calculate the norm for each objective
        norm = F_max - F_min

        # normalized distance to the worst solution
        self.pseudo_weights = ((F_max - F) / norm)

        # normalize weights to sum up to one
        self.pseudo_weights = self.pseudo_weights / np.sum(self.pseudo_weights, axis=1)[:, None]

        # normalize the provided weights as well
        weights_norm = self.weights / np.sum(self.weights)

        # search for the closest individual having this pseudo weights
        I = np.argmin(np.sum(np.abs(self.pseudo_weights - weights_norm), axis=1))

        return I


if __name__ == '__main__':
    pw = PseudoWeights(np.array([0.1, 0.4]))

    pw.do(np.random.rand(100, 2))
