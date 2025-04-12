from pymoo.algorithms.base.bracket import BracketSearch
from pymoo.core.individual import Individual
from pymoo.core.population import Population


def quadr_interp_equ(xa, fa, xb, fb, xc, fc):
    g1 = (fc - fa) / (xc - xa)
    g2 = ((fb - fc) / (xb - xc) - g1) * (xb - xa)
    xd = 0.5 * ((xa - xb) - g1 / g2)
    return xd


def quadr_interp(a, b, c):
    return Individual(X=quadr_interp_equ(a.X, a.F, b.X, b.F, c.X, c.F))


class QuadraticInterpolationSearch(BracketSearch):

    def __init__(self, a=None, b=None, **kwargs):
        """

        7.1.2 Quadratic Interpolation Search
        http://www.mathcs.emory.edu/~haber/math315/chap7.pdf

        Parameters
        ----------
        a
        b
        kwargs
        """
        super().__init__(a, b, **kwargs)

    def _initialize_infill(self):
        super()._initialize_infill()
        a, b = self.a, self.b

        # set c to be directly in the middle between the two brackets
        c = Individual(X=(b.X - a.X) / 2)

        # create a population with all three individuals
        pop = Population.create(a, b, c)

        return pop

    def _advance(self, **kwargs):

        # all the elements in the interval
        a, b, c = self.pop

        # if this is the case then the function is not convex (which means U shaped)
        if c.F[0] >= a.F[0] or c.F[0] >= b.F[0]:

            # choose the left side if a smaller than b, or the right side otherwise
            if a.F[0] <= b.F[0]:
                a = c
            else:
                b = c

            c = Individual(X=(b.X - a.X) / 2)
            self.evaluator.eval(self.problem, c, algorithm=self)
            self.infills = c

        else:

            d = quadr_interp(a, b, c)
            self.evaluator.eval(self.problem, d, algorithm=self)
            self.infills = d

            # swap c and d -> make sure d is always on the right of c -> a, c, d, b
            if c.X[0] > d.X[0]:
                c, d = d, c

            # if c is better than d, then d becomes the new right bound
            if c.F[0] <= d.F[0]:
                b = d

            # if d is better than c, then c becomes the new left bound and d the new right bound
            else:
                a, c = c, d

        self.pop = Population.create(a, b, c)
