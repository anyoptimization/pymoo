from pymoo.model.crossover import Crossover


class NoCrossover(Crossover):

    def __init__(self):
        super().__init__(1, 1, 0.0)

    def do(self, problem, pop, parents, **kwargs):
        return pop.new("X", pop.get("X"))
