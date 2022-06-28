import pymoo.gradient.toolbox as anp
import numpy as np


from pymoo.core.problem import Problem


class SYMPARTRotated(Problem):
    """
    The SYM-PART test problem proposed in [1].

    Parameters:
    -----------
    length: the length of each line (i.e., each Pareto subsets), default is 1.
    v_dist: vertical distance between the centers of two adjacent lines, default is 10.
    h_dist: horizontal distance between the centers of two adjacent lines, default is 10.
    angle: the angle to rotate the equivalent Pareto subsets counterclockwisely.
        When set to a negative value, Pareto subsets are rotated clockwisely.

    References:
    ----------
    [1] G. Rudolph, B. Naujoks, and M. Preuss, “Capabilities of EMOA to detect and preserve equivalent Pareto subsets”
    """

    def __init__(self, length=1, v_dist=10, h_dist=10, angle=np.pi / 4):
        self.a = length
        self.b = v_dist
        self.c = h_dist
        self.w = angle

        # Calculate the inverted rotation matrix, store for fitness evaluation
        self.IRM = np.array([
            [np.cos(self.w), np.sin(self.w)],
            [-np.sin(self.w), np.cos(self.w)]])

        r = max(self.b, self.c)
        xl = np.full(2, -10 * r)
        xu = np.full(2, 10 * r)

        super().__init__(n_var=2, n_obj=2, vtype=float, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        if self.w == 0:
            X1 = X[:, 0]
            X2 = X[:, 1]
        else:
            # If rotated, we rotate it back by applying the inverted rotation matrix to X
            Y = anp.array([anp.matmul(self.IRM, x) for x in X])
            X1 = Y[:, 0]
            X2 = Y[:, 1]

        a, b, c = self.a, self.b, self.c
        t1_hat = anp.sign(X1) * anp.ceil((anp.abs(X1) - a - c / 2) / (2 * a + c))
        t2_hat = anp.sign(X2) * anp.ceil((anp.abs(X2) - b / 2) / b)
        one = anp.ones(len(X))
        t1 = anp.sign(t1_hat) * anp.min(anp.vstack((anp.abs(t1_hat), one)), axis=0)
        t2 = anp.sign(t2_hat) * anp.min(anp.vstack((anp.abs(t2_hat), one)), axis=0)

        p1 = X1 - t1 * c
        p2 = X2 - t2 * b

        f1 = (p1 + a) ** 2 + p2 ** 2
        f2 = (p1 - a) ** 2 + p2 ** 2
        out["F"] = anp.vstack((f1, f2)).T

    def _calc_pareto_set(self, n_pareto_points=500):
        # The SYM-PART test problem has 9 equivalent Pareto subsets.
        h = int(n_pareto_points / 9)
        PS = np.zeros((h * 9, self.n_var))
        cnt = 0
        for row in [-1, 0, 1]:
            for col in [1, 0, -1]:
                X1 = np.linspace(row * self.c - self.a, row * self.c + self.a, h)
                X2 = np.tile(col * self.b, h)
                PS[cnt * h:cnt * h + h, :] = np.vstack((X1, X2)).T
                cnt = cnt + 1
        if self.w != 0:
            # If rotated, we apply the rotation matrix to PS
            # Calculate the rotation matrix
            RM = np.array([
                [np.cos(self.w), -np.sin(self.w)],
                [np.sin(self.w), np.cos(self.w)]
            ])
            PS = np.array([np.matmul(RM, x) for x in PS])
        return PS

    def _calc_pareto_front(self, n_pareto_points=500):
        PS = self.pareto_set(n_pareto_points)
        return self.evaluate(PS, return_values_of=["F"])


class SYMPART(SYMPARTRotated):
    def __init__(self, length=1, v_dist=10, h_dist=10):
        super().__init__(length, v_dist, h_dist, 0)
