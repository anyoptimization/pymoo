from pymoo.algorithms.convex.base import GradientBasedAlgorithm


class GradientDescent(GradientBasedAlgorithm):

    def __init__(self, X, learning_rate=0.005, **kwargs) -> None:
        super().__init__(X, **kwargs)
        self.learning_rate = learning_rate

    def restart(self):
        self.learning_rate /= 2

    def apply(self):
        self.X = self.X - self.learning_rate * self.dX
