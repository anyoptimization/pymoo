from pymoo.model.algorithm import Algorithm
from pymoo.model.individual import Individual
from pymoo.model.population import Population


class GoldenSectionSearch(Algorithm):

    def __init__(self, **kwargs):
        """



        Parameters
        ----------
        kwargs
        """
        super().__init__(**kwargs)
        self.a, self.left, self.right, self.b = None, None, None, None
        self.R = (5 ** 0.5 - 1) / 2

    def setup(self, problem, **kwargs):
        super().setup(problem, **kwargs)
        msg = "Only problems with one variable, one objective and no constraints can be solved.!"
        assert problem.n_var == 1 and not problem.has_constraints() and problem.n_obj == 1, msg

    def _initialize(self):

        # the boundaries of the problem for initialization
        xl, xu = self.problem.bounds()

        # the golden ratio (precomputed constant)
        R = self.R

        # a and b always represents the boundaries
        a, b = Individual(X=xl), Individual(X=xu)

        # create the left and right in the interval itself
        left, right = Individual(X=xu - R * (xu - xl)), Individual(X=xl + R * (xu - xl))

        # create a population with all four individuals
        pop = Population.create(a, left, right, b)

        # evaluate all the points
        self.evaluator.eval(self.problem, pop, algorithm=self)
        self.pop = pop

    def _next(self):

        # all the elements in the interval
        a, left, right, b = self.pop

        # the golden ratio (precomputed constant)
        R = self.R

        # if the left solution is better than the right
        if left.F[0] < right.F[0]:

            # make the right to be the new right bound and the left becomes the right
            a, b = a, right
            right = left

            # create a new left individual and evaluate
            left = Individual(X=b.X - R * (b.X - a.X))
            self.evaluator.eval(self.problem, Population.create(left), algorithm=self)

        # if the right solution is better than the left
        else:

            # make the left to be the new left bound and the right becomes the left
            a, b = left, b
            left = right

            # create a new right individual and evaluate
            right = Individual(X=a.X + R * (b.X - a.X))
            self.evaluator.eval(self.problem, Population.create(right), algorithm=self)

        # update the population with all the four individuals
        self.pop = Population.create(a, left, right, b)

