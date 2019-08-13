from pymoo.model.decomposition import Decomposition
from pymoo.util.function_loader import load_function


class PerpendicularDistance(Decomposition):

    def __init__(self, theta=5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.theta = theta

    def _do(self, F, weights, **kwargs):
        _, d2 = load_function("calc_distance_to_weights")(F, weights)
        return d2
