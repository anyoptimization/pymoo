from pymoo.model.algorithm import Algorithm
from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere


class GoldenSectionSearch(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.a, self.b = None, None
        self.ind_1, self.ind_2 = None, None
        self.R = (5 ** 0.5 - 1) / 2

    def setup(self, problem, **kwargs):
        super().setup(problem, **kwargs)

        if problem.n_var > 1 or problem.n_constr > 0 or problem.n_obj > 1:
            raise Exception("Only problems with one variable, one objective and no constraints"
                            "can be solved by the Golden Section Search!")

    def _initialize(self):
        xl, xu = self.problem.bounds()
        R = self.R
        d = R * (xu - xl)

        a = Individual(X=xl)
        b = Individual(X=xu)

        left = Individual(X=xu - d)
        right = Individual(X=xl + d)

        pop = Population.create(a, left, right, b)

        self.evaluator.eval(self.problem, pop, algorithm=self)
        self.pop = pop

    def _next(self):
        a, left, right, b = self.pop
        R = self.R

        if left.F[0] < right.F[0]:

            a, b = a, right
            right = left

            left = Individual(X=b.X - R * (b.X - a.X))
            self.evaluator.eval(self.problem, Population.create(left), algorithm=self)

        else:

            a, b = left, b
            left = right

            right = Individual(X=a.X + R * (b.X - a.X))
            self.evaluator.eval(self.problem, Population.create(right), algorithm=self)

        self.pop = Population.create(a, left, right, b)

