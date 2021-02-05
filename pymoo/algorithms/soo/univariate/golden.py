from pymoo.algorithms.base.bracket import BracketSearch
from pymoo.model.individual import Individual
from pymoo.model.population import Population


class GoldenSectionSearch(BracketSearch):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.left, self.right = None, None
        self.R = (5 ** 0.5 - 1) / 2

    def _initialize(self):
        super()._initialize()
        a, b = self.a, self.b

        # the golden ratio (precomputed constant)
        R = self.R

        # create the left and right in the interval itself
        c, d = Individual(X=b.X - R * (b.X - a.X)), Individual(X=a.X + R * (b.X - a.X))

        # create a population with all four individuals
        pop = Population.create(a, c, d, b)

        self.pop, self.infill = pop, pop

        return pop

    def step(self):

        # all the elements in the interval
        a, c, d, b = self.pop

        # the golden ratio (precomputed constant)
        R = self.R

        # if the left solution is better than the right
        if c.F[0] < d.F[0]:

            # make the right to be the new right bound and the left becomes the right
            a, b = a, d
            d = c

            # create a new left individual and evaluate
            c = Individual(X=b.X - R * (b.X - a.X))
            self.evaluator.eval(self.problem, c, algorithm=self)
            self.infill = c

        # if the right solution is better than the left
        else:

            # make the left to be the new left bound and the right becomes the left
            a, b = c, b
            c = d

            # create a new right individual and evaluate
            d = Individual(X=a.X + R * (b.X - a.X))
            self.evaluator.eval(self.problem, d, algorithm=self)
            self.infill = d

        # update the population with all the four individuals
        self.pop = Population.create(a, c, d, b)
