from pymoo.model.crossover import Crossover


class DifferentialEvolutionCrossover(Crossover):

    def __init__(self, weight=0.8):
        super().__init__(3, 1)
        self.weight = weight

    def _do(self, problem, pop, parents, **kwargs):
        X = pop.get("X")[parents.T]
        _X = X[0] + self.weight * (X[1] - X[2])
        return pop.new("X", _X)
