from pymoo.core.algorithm import Algorithm
from pymoo.core.individual import Individual


class BracketSearch(Algorithm):

    def __init__(self, a=None, b=None, **kwargs):
        super().__init__(**kwargs)
        self.a, self.b = a, b

    def _setup(self, problem, a=None, b=None, **kwargs):
        msg = "Only problems with one variable, one objective and no constraints can be solved!"
        assert problem.n_var == 1 and not problem.has_constraints() and problem.n_obj == 1, msg
        self.a, self.b = a, b

    def _initialize_infill(self):

        # the boundaries of the problem for initialization
        xl, xu = self.problem.bounds()

        # take care of the lower bound - make sure it is an individual and make sure it is evaluated
        if self.a is None:
            assert xl is not None, "Either provide a left bound or set the xl attribute in the problem."
            self.a = Individual(X=xl)

        if self.a.F is None:
            self.evaluator.eval(self.problem, self.a, algorithm=self)

        # take care of the upper bound - make sure it is an individual and make sure it is evaluated
        if self.b is None:
            assert xl is not None, "Either provide a right bound or set the xu attribute in the problem."
            self.b = Individual(X=xu)

        if self.b.F is None:
            self.evaluator.eval(self.problem, self.b, algorithm=self)

        assert self.a.X[0] <= self.b.X[0], "The left bound must be smaller than the left bound!"

