import autograd.numpy as anp

from pymoo.model.problem import Problem


class Truss2D(Problem):

    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_constr=1, type_var=anp.double)

        self.Amax = 0.01
        self.Smax = 1e5

        self.xl = anp.array([0.0, 0.0, 1.0])
        self.xu = anp.array([self.Amax, self.Amax, 3.0])

    def _evaluate(self, x, out, *args, **kwargs):

        # variable names for convenient access
        x1 = x[:, 0]
        x2 = x[:, 1]
        y = x[:, 2]

        # first objectives
        f1 = x1 * anp.sqrt(16 + anp.square(y)) + x2 * anp.sqrt((1 + anp.square(y)))

        # measure which are needed for the second objective
        sigma_ac = 20 * anp.sqrt(16 + anp.square(y)) / (y * x1)
        sigma_bc = 80 * anp.sqrt(1 + anp.square(y)) / (y * x2)

        # take the max
        f2 = anp.max(anp.column_stack((sigma_ac, sigma_bc)), axis=1)

        # define a constraint
        g1 = f2 - self.Smax

        out["F"] = anp.column_stack([f1, f2])
        out["G"] = g1
