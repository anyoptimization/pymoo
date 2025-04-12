import pymoo.gradient.toolbox as anp
import numpy as np

from pymoo.core.problem import Problem


class Truss2D(Problem):

    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_ieq_constr=1, vtype=float)

        self.Amax = 0.01
        self.Smax = 1e5

        self.xl = np.array([0.0, 0.0, 1.0])
        self.xu = np.array([self.Amax, self.Amax, 3.0])

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

    def _calc_pareto_front(self, *args, **kwargs):

        T = 2 * np.sqrt(5) * self.Amax

        # Part A - before transition point T

        f1 = np.linspace(400 / self.Smax, T, 1000)
        f2 = 400 / f1
        part_a = np.column_stack([f1, f2])

        # Part B - after transition point T

        def calc_y(V):
            return np.sqrt(3200 * V**2 + 40 * V * np.sqrt(6400 * V**2 - 12) - 4)

        def calc_SV(y):
            return (4 + y**2) / (0.01 * y)

        def calc_x1(y):
            return 0.0025 * np.sqrt((16 + y**2) / (1 + y**2))

        def calc_S(V):
            y = calc_y(V)
            SV = calc_SV(y)
            S = SV / V
            return S

        y = 3
        x1 = calc_x1(y)
        x2 = 0.01

        V_min = T
        V_max = self.evaluate(np.array([x1, x2, y]), return_values_of=["F"])[0]

        f1 = np.linspace(V_min, V_max, 100)
        f2 = calc_S(f1)
        part_b = np.column_stack([f1, f2])

        pf = np.row_stack([part_a, part_b])

        return pf


