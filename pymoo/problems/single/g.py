"""
The G problems were originally defined at a CEC competition in 2006:
Liang, Jing J., Thomas Philip Runarsson, Efrén Mezura-Montes, Maurice Clerc, Ponnuthurai Nagaratnam Suganthan, Carlos A. Coello Coello, and Kalyanmoy Deb.
Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization.
"""

import math

import numpy as np

import pymoo.gradient.toolbox as anp
from pymoo.core.problem import Problem
from pymoo.util.misc import at_least_2d_array


class G(Problem):

    def _calc_pareto_front(self):
        ps = at_least_2d_array(self._calc_pareto_set(), extend_as="r")
        return self.evaluate(ps, return_as_dictionary=True)["F"].min(axis=0)


class G1(G):
    def __init__(self):
        n_var = 13
        xl = np.zeros(n_var, dtype=float)
        xu = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 1], dtype=float)
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=9, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        x1 = x[:, 0: 4]
        x2 = x[:, 4: 13]

        f = 5 * x1.sum(axis=1) - 5 * (x1 ** 2).sum(axis=1) - x2.sum(axis=1)

        # Constraints
        g1 = 2 * x[:, 0] + 2 * x[:, 1] + x[:, 9] + x[:, 10] - 10
        g2 = 2 * x[:, 0] + 2 * x[:, 2] + x[:, 9] + x[:, 11] - 10
        g3 = 2 * x[:, 1] + 2 * x[:, 2] + x[:, 10] + x[:, 11] - 10
        g4 = -8 * x[:, 0] + x[:, 9]
        g5 = -8 * x[:, 1] + x[:, 10]
        g6 = -8 * x[:, 2] + x[:, 11]
        g7 = -2 * x[:, 3] - x[:, 4] + x[:, 9]
        g8 = -2 * x[:, 5] - x[:, 6] + x[:, 10]
        g9 = -2 * x[:, 7] - x[:, 8] + x[:, 11]

        out["F"] = f
        out["G"] = [g1, g2, g3, g4, g5, g6, g7, g8, g9]

    def _calc_pareto_front(self):
        return -15.0

    def _calc_pareto_set(self):
        return np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1], dtype=float)


class G2(G):
    def __init__(self, n_var=20):
        xl = np.full(n_var, 1e-16)
        xu = 10 * np.ones(n_var)
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=2, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        a = anp.sum(anp.cos(x) ** 4, axis=1)
        b = 2 * anp.prod(anp.cos(x) ** 2, axis=1)

        sum_jx = 0.0
        for j in range(self.n_var):
            sum_jx = sum_jx + (j + 1) * x[:, j] ** 2

        c = anp.sqrt(sum_jx) + (sum_jx == 0) * 1e-64
        f = -anp.absolute((a - b) / c)

        # Constraints
        g1 = -anp.prod(x, 1) + 0.75
        g2 = anp.sum(x, axis=1) - 7.5 * self.n_var

        out["F"] = f
        out["G"] = anp.column_stack([g1, g2])

    def _calc_pareto_set(self):
        if self.n_var == 2:
            return np.array([1.600859, 0.4684985])
        elif self.n_var == 10:
            return np.array([3.1238477, 3.0690696, 3.0139085, 2.9572856, 1.4654789, 0.3684877, 0.3633289, 0.3592627,
                             0.3547453, 0.3510025])

        # the version of the paper with 20 variables
        elif self.n_var == 20:
            return np.array([3.16246061572185, 3.12833142812967, 3.09479212988791, 3.06145059523469, 3.02792915885555,
                             2.99382606701730, 2.95866871765285, 2.92184227312450, 0.49482511456933, 0.48835711005490,
                             0.48231642711865, 0.47664475092742, 0.47129550835493, 0.46623099264167, 0.46142004984199,
                             0.45683664767217, 0.45245876903267, 0.44826762241853, 0.44424700958760, 0.44038285956317])


class G3(G):

    def __init__(self, n_var=10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var=n_var, n_obj=1, n_eq_constr=1, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = - anp.sqrt(self.n_var) ** self.n_var * anp.prod(x, axis=1)
        h = anp.sum(x ** 2, axis=1) - 1

        out["F"] = f
        out["H"] = h

    def _calc_pareto_set(self):
        return np.full(self.n_var, 1 / np.sqrt(self.n_var))


class G4(G):

    def __init__(self):
        xl = np.array([78, 33, 27, 27, 27], dtype=float)
        xu = np.array([102, 45, 45, 45, 45], dtype=float)
        super().__init__(n_var=5, n_obj=1, n_ieq_constr=6, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = 5.3578547 * x[:, 2] ** 2 + 0.8356891 * x[:, 0] * x[:, 4] + 37.293239 * x[:, 0] - 40792.141

        # Constraints
        u = 85.334407 + 0.0056858 * x[:, 1] * x[:, 4] + 0.0006262 * x[:, 0] * x[:, 3] - 0.0022053 * x[:, 2] * x[:, 4]
        g1 = -u
        g2 = u - 92
        v = 80.51249 + 0.0071317 * x[:, 1] * x[:, 4] + 0.0029955 * x[:, 0] * x[:, 1] + 0.0021813 * x[:, 2] ** 2
        g3 = -v + 90
        g4 = v - 110
        w = 9.300961 + 0.0047026 * x[:, 2] * x[:, 4] + 0.0012547 * x[:, 0] * x[:, 2] + 0.0019085 * x[:, 2] * x[:, 3]
        g5 = -w + 20
        g6 = w - 25

        out["F"] = f
        out["G"] = anp.column_stack([g1, g2, g3, g4, g5, g6])

    def _calc_pareto_set(self):
        return [78, 33, 29.9952560256815985, 45, 36.7758129057882073]


class G5(G):

    def __init__(self):
        xl = np.array([0, 0, -0.55, -0.55], dtype=float)
        xu = np.array([1200, 1200, 0.55, 0.55], dtype=float)
        super().__init__(n_var=4, n_obj=1, n_ieq_constr=2, n_eq_constr=3, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = 3 * x[:, 0] + (10 ** -6) * x[:, 0] ** 3 + 2 * x[:, 1] + (2 * 10 ** (-6)) / 3 * x[:, 1] ** 3

        # Inequality Constraints
        g1 = x[:, 2] - x[:, 3] - 0.55
        g2 = x[:, 3] - x[:, 2] - 0.55

        # Equality Constraints
        h1 = 1000 * anp.sin(-x[:, 2] - 0.25) + 1000 * anp.sin(-x[:, 3] - 0.25) + 894.8 - x[:, 0]
        h2 = 1000 * anp.sin(x[:, 2] - 0.25) + 1000 * anp.sin(x[:, 2] - x[:, 3] - 0.25) + 894.8 - x[:, 1]
        h3 = 1000 * anp.sin(x[:, 3] - 0.25) + 1000 * anp.sin(x[:, 3] - x[:, 2] - 0.25) + 1294.8

        out["F"] = f
        out["G"] = anp.column_stack([g1, g2])
        out["H"] = anp.column_stack([h1, h2, h3])

    def _calc_pareto_set(self):
        return [679.94531748791177961,
                1026.06713513571594376,
                0.11887636617838561,
                -0.39623355240329272]


class G6(G):

    def __init__(self):
        xl = np.array([13, 0], dtype=float)
        xu = np.array([100, 100], dtype=float)
        super().__init__(n_var=2, n_obj=1, n_ieq_constr=2, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = (x[:, 0] - 10) ** 3 + (x[:, 1] - 20) ** 3

        # Constraints
        g1 = -(x[:, 0] - 5) ** 2 - (x[:, 1] - 5) ** 2 + 100
        g2 = (x[:, 0] - 6) ** 2 + (x[:, 1] - 5) ** 2 - 82.81

        out["F"] = f
        out["G"] = anp.column_stack([g1, g2])

    def _calc_pareto_set(self):
        return np.array([14.095, 5 - np.sqrt(100 - (14.095 - 5) ** 2)])


class G7(G):

    def __init__(self):
        n_var = 10
        xl = -10 * np.ones(n_var)
        xu = 10 * np.ones(n_var)
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=8, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 0] * x[:, 1] - 14 * x[:, 0] - 16 * x[:, 1] + (x[:, 2] - 10) ** 2 \
            + 4 * (x[:, 3] - 5) ** 2 + (x[:, 4] - 3) ** 2 + 2 * (x[:, 5] - 1) ** 2 + 5 * x[:, 6] ** 2 \
            + 7 * (x[:, 7] - 11) ** 2 + 2 * (x[:, 8] - 10) ** 2 + (x[:, 9] - 7) ** 2 + 45

        # Constraints
        g1 = 4 * x[:, 0] + 5 * x[:, 1] - 3 * x[:, 6] + 9 * x[:, 7] - 105
        g2 = 10 * x[:, 0] - 8 * x[:, 1] - 17 * x[:, 6] + 2 * x[:, 7]
        g3 = -8 * x[:, 0] + 2 * x[:, 1] + 5 * x[:, 8] - 2 * x[:, 9] - 12
        g4 = 3 * (x[:, 0] - 2) ** 2 + 4 * (x[:, 1] - 3) ** 2 + 2 * x[:, 2] ** 2 - 7 * x[:, 3] - 120
        g5 = 5 * x[:, 0] ** 2 + 8 * x[:, 1] + (x[:, 2] - 6) ** 2 - 2 * x[:, 3] - 40
        g6 = x[:, 0] ** 2 + 2 * (x[:, 1] - 2) ** 2 - 2 * x[:, 0] * x[:, 1] + 14 * x[:, 4] - 6 * x[:, 5]
        g7 = 0.5 * (x[:, 0] - 8) ** 2 + 2 * (x[:, 1] - 4) ** 2 + 3 * x[:, 4] ** 2 - x[:, 5] - 30
        g8 = -3 * x[:, 0] + 6 * x[:, 1] + 12 * (x[:, 8] - 8) ** 2 - 7 * x[:, 9]

        out["F"] = f
        out["G"] = anp.column_stack([g1, g2, g3, g4, g5, g6, g7, g8])

    def _calc_pareto_set(self):
        return [2.171997834812,
                2.363679362798,
                8.773925117415,
                5.095984215855,
                0.990655966387,
                1.430578427576,
                1.321647038816,
                9.828728107011,
                8.280094195305,
                8.375923511901]


class G8(G):

    def __init__(self):
        n_var = 2
        xl = np.full(n_var, 0.00001)
        xu = np.full(n_var, 10.0)
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=2, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = -(anp.sin(2 * math.pi * x[:, 0]) ** 3 * anp.sin(2 * math.pi * x[:, 1])) / (
                x[:, 0] ** 3 * (x[:, 0] + x[:, 1]))

        # Constraints
        g1 = x[:, 0] ** 2 - x[:, 1] + 1
        g2 = 1 - x[:, 0] + (x[:, 1] - 4) ** 2

        out["F"] = f
        out["G"] = anp.column_stack([g1, g2])

    def _calc_pareto_set(self):
        return [1.22797135260752599, 4.24537336612274885]


class G9(G):

    def __init__(self):
        n_var = 7
        xl = np.full(n_var, -10.0)
        xu = np.full(n_var, +10.0)
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=4, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = (x[:, 0] - 10) ** 2 + 5 * (x[:, 1] - 12) ** 2 + x[:, 2] ** 4 \
            + 3 * (x[:, 3] - 11) ** 2 + 10 * x[:, 4] ** 6 + 7 * x[:, 5] ** 2 \
            + x[:, 6] ** 4 - 4 * x[:, 5] * x[:, 6] - 10 * x[:, 5] - 8 * x[:, 6]

        # Constraints
        v1 = 2 * x[:, 0] ** 2
        v2 = x[:, 1] ** 2
        g1 = v1 + 3 * v2 ** 2 + x[:, 2] + 4 * x[:, 3] ** 2 + 5 * x[:, 4] - 127
        g2 = 7 * x[:, 0] + 3 * x[:, 1] + 10 * x[:, 2] ** 2 + x[:, 3] - x[:, 4] - 282
        g3 = 23 * x[:, 0] + v2 + 6 * x[:, 5] ** 2 - 8 * x[:, 6] - 196
        g4 = 2 * v1 + v2 - 3 * x[:, 0] * x[:, 1] + 2 * x[:, 2] ** 2 + 5. * x[:, 5] - 11 * x[:, 6]

        out["F"] = f[:, None]
        out["G"] = anp.column_stack([g1, g2, g3, g4])

    def _calc_pareto_set(self):
        # return [2.33049935147405174, 1.95137236847114592, -0.477541399510615805, 4.36572624923625874,
        #         -0.624486959100388983, 1.03813099410962173, 1.5942266780671519]
        return [
            2.33049949323300210,
            1.95137239646596039,
            -0.47754041766198602,
            4.36572612852776931,
            -0.62448707583702823,
            1.03813092302119347,
            1.59422663221959926]


class G10(G):

    def __init__(self):
        xl = np.array([100, 1000, 1000, 10, 10, 10, 10, 10], dtype=float)
        xu = np.array([10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000], dtype=float)
        super().__init__(n_var=8, n_obj=1, n_ieq_constr=6, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = x[:, 0] + x[:, 1] + x[:, 2]

        # Constraints
        g1 = -1 + 0.0025 * (x[:, 3] + x[:, 5])
        g2 = -1 + 0.0025 * (-x[:, 3] + x[:, 4] + x[:, 6])
        g3 = -1 + 0.01 * (-x[:, 4] + x[:, 7])
        g4 = 100 * x[:, 0] - x[:, 0] * x[:, 5] + 833.33252 * x[:, 3] - 83333.333
        g5 = x[:, 1] * x[:, 3] - x[:, 1] * x[:, 6] - 1250 * x[:, 3] + 1250 * x[:, 4]
        g6 = x[:, 2] * x[:, 4] - x[:, 2] * x[:, 7] - 2500. * x[:, 4] + 1250000

        out["F"] = f
        out["G"] = anp.column_stack([g1, g2, g3, g4, g5, g6])

    def _calc_pareto_set(self):
        return [579.29340269759155,
                1359.97691009458777,
                5109.97770901501008,
                182.01659025342749,
                295.60089166064103,
                217.98340973906758,
                286.41569858295981,
                395.60089165381908]


class G11(G):

    def __init__(self):
        xl = np.array([-1, -1], dtype=float)
        xu = np.array([1, 1], dtype=float)
        super().__init__(n_var=2, n_obj=1, n_ieq_constr=1, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = x[:, 0] ** 2 + (x[:, 1] - 1) ** 2
        g = x[:, 1] - x[:, 0] ** 2

        out["F"] = f
        out["G"] = g

    def _calc_pareto_set(self):
        return [-np.sqrt(0.5), 0.5]


class G12(G):

    def __init__(self):
        xl = np.full(3, 0.0)
        xu = np.full(3, 10.0)
        super().__init__(n_var=3, n_obj=1, n_ieq_constr=1, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = -1 + 0.01 * ((x[:, 0] - 5) ** 2 + (x[:, 1] - 5) ** 2 + (x[:, 2] - 5) ** 2)

        g = anp.full(len(x), anp.inf)
        for i in range(1, 10):
            for j in range(1, 10):
                for k in range(1, 10):
                    g = anp.minimum(g, (x[:, 0] - i) ** 2 + (x[:, 1] - j) ** 2 + (x[:, 2] - k) ** 2 - 0.0625)

        out["F"] = f
        out["G"] = g

    def _calc_pareto_set(self):
        return [5.0, 5.0, 5.0]


class G13(G):

    def __init__(self):
        xl = np.array([-2.3, -2.3, -3.2, -3.2, -3.2])
        xu = np.array([+2.3, +2.3, +3.2, +3.2, +3.2])
        super().__init__(n_var=5, n_obj=1, n_eq_constr=3, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = anp.exp(x[:, 0] * x[:, 1] * x[:, 2] * x[:, 3] * x[:, 4])

        h1 = x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2 + x[:, 3] ** 2 + x[:, 4] ** 2 - 10
        h2 = x[:, 1] * x[:, 2] - 5 * x[:, 3] * x[:, 4]
        h3 = x[:, 0] ** 3 + x[:, 1] ** 3 + 1

        out["F"] = f
        out["H"] = anp.column_stack([h1, h2, h3])

    def _calc_pareto_set(self):
        opt = np.array([-1.7171435947203, 1.5957097321519, 1.8272456947885, -0.7636422812896, -0.7636439027742])
        ps = [opt,
              np.array([opt[0], opt[1], -opt[2], -opt[3], +opt[4]]),
              np.array([opt[0], opt[1], -opt[2], +opt[3], -opt[4]]),
              np.array([opt[0], opt[1], +opt[2], -opt[3], -opt[4]]),
              np.array([opt[0], opt[1], -opt[2], +opt[3], -opt[4]]),
              np.array([opt[0], opt[1], -opt[2], -opt[3], +opt[4]])
              ]
        return np.vstack(ps)


class G14(G):

    def __init__(self):
        xl = np.full(10, 1e-06)
        xu = np.full(10, 10.0)
        super().__init__(n_var=10, n_obj=1, n_eq_constr=3, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        v = anp.array([-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179])
        y = anp.log(x / anp.sum(x, axis=1, keepdims=True))
        f = anp.sum(x * (v + y), axis=1)

        h1 = x[:, 0] + 2 * x[:, 1] + 2 * x[:, 2] + x[:, 5] + x[:, 9] - 2
        h2 = x[:, 3] + 2 * x[:, 4] + x[:, 5] + x[:, 6] - 1
        h3 = x[:, 2] + x[:, 6] + x[:, 7] + 2 * x[:, 8] + x[:, 9] - 1

        out["F"] = f
        out["H"] = anp.column_stack([h1, h2, h3])

    def _calc_pareto_set(self):
        return [0.0406684113216282, 0.147721240492452, 0.783205732104114, 0.00141433931889084, 0.485293636780388,
                0.000693183051556082, 0.0274052040687766, 0.0179509660214818, 0.0373268186859717, 0.0968844604336845]


class G15(G):

    def __init__(self):
        n_var = 3
        xl = np.full(n_var, 0.0)
        xu = np.full(n_var, 10.0)
        super().__init__(n_var=n_var, n_obj=1, n_eq_constr=2, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = 1000 - (x[:, 0] ** 2) - 2 * x[:, 1] ** 2 - x[:, 2] ** 2 - x[:, 0] * x[:, 1] - x[:, 0] * x[:, 2]

        h1 = x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2 - 25
        h2 = 8 * x[:, 0] + 14 * x[:, 1] + 7 * x[:, 2] - 56

        out["F"] = f
        out["H"] = anp.column_stack([h1, h2])

    def _calc_pareto_set(self):
        return [3.51212812611795133, 0.216987510429556135, 3.55217854929179921]


class G16(G):

    def __init__(self):
        xl = np.array([704.4148, 68.6, 0, 193, 25], dtype=float)
        xu = np.array([906.3855, 288.88, 134.75, 287.0966, 84.1988], dtype=float)
        super().__init__(n_var=5, n_obj=1, n_ieq_constr=38, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        y1 = x[:, 1] + x[:, 2] + 41.6
        c1 = 0.024 * x[:, 3] - 4.62
        y2 = (12.5 / c1) + 12
        c2 = 0.0003535 * (x[:, 0] ** 2) + 0.5311 * x[:, 0] + 0.08705 * y2 * x[:, 0]
        c3 = 0.052 * x[:, 0] + 78 + 0.002377 * y2 * x[:, 0]
        y3 = c2 / c3
        y4 = 19 * y3
        c4 = 0.04782 * (x[:, 0] - y3) + (0.1956 * (x[:, 0] - y3) ** 2) / x[:, 1] + 0.6376 * y4 + 1.594 * y3
        c5 = 100 * x[:, 1]
        c6 = x[:, 0] - y3 - y4
        c7 = 0.950 - (c4 / c5)
        y5 = c6 * c7
        y6 = x[:, 0] - y5 - y4 - y3
        c8 = (y5 + y4) * 0.995
        y7 = c8 / y1
        y8 = c8 / 3798
        c9 = y7 - (0.0663 * (y7 / y8)) - 0.3153
        y9 = (96.82 / c9) + 0.321 * y1
        y10 = 1.29 * y5 + 1.258 * y4 + 2.29 * y3 + 1.71 * y6
        y11 = 1.71 * x[:, 0] - 0.452 * y4 + 0.580 * y3
        c10 = 12.3 / 752.3
        c11 = (1.75 * y2) * (0.995 * x[:, 0])
        c12 = (0.995 * y10) + 1998
        y12 = c10 * x[:, 0] + (c11 / c12)
        y13 = c12 - 1.75 * y2
        y14 = 3623 + 64.4 * x[:, 1] + 58.4 * x[:, 2] + 146312 / (y9 + x[:, 4])
        c13 = 0.995 * y10 + 60.8 * x[:, 1] + 48 * x[:, 3] - 0.1121 * y14 - 5095
        y15 = y13 / c13
        y16 = 148000 - 331000 * y15 + 40 * y13 - 61 * y15 * y13
        c14 = 2324 * y10 - 28740000 * y2
        y17 = 14130000 - (1328 * y10) - (531 * y11) + (c14 / c12)
        c15 = (y13 / y15) - (y13 / 0.52)
        c16 = 1.104 - 0.72 * y15
        c17 = y9 + x[:, 4]

        f = (0.000117 * y14) + 0.1365 + (0.00002358 * y13) + (0.000001502 * y16) + (0.0321 * y12) + (0.004324 * y5) + (
                0.0001 * c15 / c16) + (37.48 * (y2 / c12)) - (0.0000005843 * y17)

        g1 = (0.28 / 0.72) * y5 - y4
        g2 = x[:, 2] - 1.5 * x[:, 1]
        g3 = 3496 * (y2 / c12) - 21
        g4 = 110.6 + y1 - (62212 / c17)
        g5 = 213.1 - y1
        g6 = y1 - 405.23
        g7 = 17.505 - y2
        g8 = y2 - 1053.6667
        g9 = 11.275 - y3
        g10 = y3 - 35.03
        g11 = 214.228 - y4
        g12 = y4 - 665.585
        g13 = 7.458 - y5
        g14 = y5 - 584.463
        g15 = 0.961 - y6
        g16 = y6 - 265.916
        g17 = 1.612 - y7
        g18 = y7 - 7.046
        g19 = 0.146 - y8
        g20 = y8 - 0.222
        g21 = 107.99 - y9
        g22 = y9 - 273.366
        g23 = 922.693 - y10
        g24 = y10 - 1286.105
        g25 = 926.832 - y11
        g26 = y11 - 1444.046
        g27 = 18.766 - y12
        g28 = y12 - 537.141
        g29 = 1072.163 - y13
        g30 = y13 - 3247.039
        g31 = 8961.448 - y14
        g32 = y14 - 26844.086
        g33 = 0.063 - y15
        g34 = y15 - 0.386
        g35 = 71084.33 - y16
        g36 = -140000 + y16
        g37 = 2802713 - y17
        g38 = y17 - 12146108

        out["F"] = f
        out["G"] = anp.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16, g17, g18,
                                     g19, g20, g21, g22, g23, g24, g25, g26, g27, g28, g29, g30, g31, g32, g33, g34,
                                     g35, g36, g37, g38])

    def _calc_pareto_set(self):
        return [705.17454, 68.60000, 102.90000, 282.32493, 37.58412]


class G17(G):

    def __init__(self):
        xl = np.array([0, 0, 340, 340, -1000, 0], dtype=float)
        xu = np.array([400, 1000, 420, 420, 1000, 0.5236], dtype=float)
        super().__init__(n_var=6, n_obj=1, n_eq_constr=4, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = anp.zeros(len(x))

        x1_less_than_300 = x[:, 0] < 300
        f = f + x1_less_than_300 * 30 * x[:, 0]
        f = f + (~x1_less_than_300) * 31 * x[:, 0]

        x2_less_than_100 = x[:, 1] < 100
        x2_greater_equal_200 = x[:, 1] >= 200

        f = f + x2_less_than_100 * 28 * x[:, 1]
        f = f + x2_greater_equal_200 * 30 * x[:, 1]

        x2_between_100_and_200 = anp.logical_and(~x2_less_than_100, ~x2_greater_equal_200)
        f = f + x2_between_100_and_200 * 29 * x[:, 1]

        h1 = -x[:, 0] + 300 - ((x[:, 2] * x[:, 3]) / 131.078) * anp.cos(1.48477 - x[:, 5]) + (
                (0.90798 * x[:, 2] ** 2) / 131.078) * anp.cos(1.47588)
        h2 = -x[:, 1] - ((x[:, 2] * x[:, 3]) / 131.078) * anp.cos(1.48477 + x[:, 5]) + (
                (0.90798 * x[:, 3] ** 2) / 131.078) * anp.cos(1.47588)
        h3 = -x[:, 4] - ((x[:, 2] * x[:, 3]) / 131.078) * anp.sin(1.48477 + x[:, 5]) + (
                (0.90798 * x[:, 3] ** 2) / 131.078) * anp.sin(1.47588)
        h4 = 200 - ((x[:, 2] * x[:, 3]) / 131.078) * anp.sin(1.48477 - x[:, 5]) + (
                (0.90798 * x[:, 2] ** 2) / 131.078) * anp.sin(1.47588)

        out["F"] = f
        out["H"] = anp.column_stack([h1, h2, h3, h4])

    def _calc_pareto_set(self):
        return [201.784467214523659, 99.9999999999999005, 383.071034852773266, 420, -10.9076584514292652,
                0.0731482312084287128]


class G18(G):

    def __init__(self):
        xl = np.array([-10, -10, -10, -10, -10, -10, -10, -10, 0], dtype=float)
        xu = np.array([+10, +10, +10, +10, +10, +10, +10, +10, 20], dtype=float)
        super().__init__(n_var=9, n_obj=1, n_ieq_constr=13, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = -0.5 * (x[:, 0] * x[:, 3] - x[:, 1] * x[:, 2] + x[:, 2] * x[:, 8] - x[:, 4] * x[:, 8] + x[:, 4]
                    * x[:, 7] - x[:, 5] * x[:, 6])

        g1 = x[:, 2] ** 2 + x[:, 3] ** 2 - 1
        g2 = x[:, 8] ** 2 - 1
        g3 = x[:, 4] ** 2 + x[:, 5] ** 2 - 1
        g4 = x[:, 0] ** 2 + (x[:, 1] - x[:, 8]) ** 2 - 1
        g5 = (x[:, 0] - x[:, 4]) ** 2 + (x[:, 1] - x[:, 5]) ** 2 - 1
        g6 = (x[:, 0] - x[:, 6]) ** 2 + (x[:, 1] - x[:, 7]) ** 2 - 1
        g7 = (x[:, 2] - x[:, 4]) ** 2 + (x[:, 3] - x[:, 5]) ** 2 - 1
        g8 = (x[:, 2] - x[:, 6]) ** 2 + (x[:, 3] - x[:, 7]) ** 2 - 1
        g9 = x[:, 6] ** 2 + (x[:, 7] - x[:, 8]) ** 2 - 1
        g10 = x[:, 1] * x[:, 2] - x[:, 0] * x[:, 3]
        g11 = -x[:, 2] * x[:, 8]
        g12 = x[:, 4] * x[:, 8]
        g13 = x[:, 5] * x[:, 6] - x[:, 4] * x[:, 7]

        out["F"] = f
        out["G"] = anp.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13])

    def _calc_pareto_set(self):
        return [
            -0.9890005492667746, 0.1479118418638228, -0.6242897641574451, -0.7811841737429015, -0.9876159387318453,
            0.1504778305249072, -0.6225959783340022, -0.782543417629948, 0.0
        ]


class G19(G):

    def __init__(self):
        super().__init__(n_var=15, n_obj=1, n_ieq_constr=5, xl=np.full(15, 0.0), xu=np.full(15, 10.0), vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        aMat19 = anp.array([-16, 2, 0, 1, 0,
                            +0, -2, 0, 0.4, 2,
                            -3.5, 0, 2, 0, 0,
                            +0, -2, 0, -4, -1,
                            +0, -9, -2, 1, -2.8,
                            +2, 0, -4, 0, 0,
                            -1, -1, -1, -1, -1,
                            -1, -2, -3, -2, -1,
                            +1, 2, 3, 4, 5,
                            +1, 1, 1, 1, 1]).reshape((10, 5))

        bVec19 = anp.array([-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1])

        cMat19 = anp.array([+30, -20, -10, 32, -10,
                            -20, 39, -6, -31, 32,
                            -10, -6, 10, -6, -10,
                            +32, -31, -6, 39, -20,
                            -10, 32, -10, -20, 30]).reshape((5, 5))

        dVec19 = anp.array([4, 8, 10, 6, 2])
        eVec19 = anp.array([-15, -27, -36, -18, -12])

        f = - anp.sum(bVec19 * x[:, :10], axis=1) + 2 * anp.sum(dVec19 * x[:, 10:] * x[:, 10:] * x[:, 10:], axis=1)

        for i in range(5):
            f = f + x[:, 10 + i] * anp.sum(cMat19[i] * x[:, 10:], axis=1)

        g = []
        for j in range(5):
            _g = -2 * anp.sum(cMat19[j] * x[:, 10:], axis=1) - 3 * dVec19[j] * x[:, 10 + j] * x[:, 10 + j] \
                 - eVec19[j] + anp.sum(aMat19[:, j] * x[:, :10], axis=1)
            g.append(_g)

        out["F"] = f
        out["G"] = anp.column_stack(g)

    def _calc_pareto_set(self):
        return [
            0, 0, 3.94600628013917, 0, 3.28318162727873, 10, 0, 0, 0, 0, 0.370762125835098, 0.278454209512692,
            0.523838440499861, 0.388621589976956, 0.29815843730292
        ]


class G20(G):

    def __init__(self):
        n_var = 24
        xl = np.full(n_var, 0.0)
        xu = np.full(n_var, 10.0)
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=6, n_eq_constr=14, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        a = anp.array([0.0693, 0.0577, 0.05, 0.2,
                       0.26, 0.55, 0.06, 0.1, 0.12,
                       0.18, 0.1, 0.09, 0.0693, 0.0577,
                       0.05, 0.2, 0.26, 0.55, 0.06,
                       0.1, 0.12, 0.18, 0.1, 0.09])

        f = anp.sum(a * x, axis=1)

        e = anp.array([0.1, 0.3, 0.4, 0.3, 0.6, 0.3])

        b = anp.array([44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94,
                       133.425, 82.507, 46.07, 60.097, 44.094, 58.12, 58.12,
                       137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507, 46.07, 60.097])

        cVec = anp.array([123.7, 31.7, 45.7, 14.7, 84.7, 27.7, 49.7, 7.1, 2.1, 17.7, 0.85, 0.64])
        d = anp.array([31.244, 36.12, 34.784, 92.7, 82.7, 91.6, 56.708, 82.7, 80.8, 64.517, 49.4, 49.1])
        k = 0.7302 * 530 * (14.7 / 40)
        sumX = anp.sum(x, axis=1)

        g123 = [(x[:, i] + x[:, i + 12]) / (sumX + e[i]) for i in range(3)]
        g456 = [(x[:, i + 3] + x[:, i + 15]) / (sumX + e[i]) for i in range(3, 6)]

        h = []
        for i in range(12):
            h1 = x[:, i + 12] / (b[i + 12] * anp.sum((x / b)[:, 12:], axis=1))
            h2 = cVec[i] * x[:, i] / (40 * b[i] * anp.sum((x / b)[:, :12], axis=1))
            h.append(h1 - h2)

        h13 = sumX - 1
        h14 = anp.sum((x[:, :12] / d), axis=1) + k * anp.sum((x / b)[:, 12:], axis=1) - 1.671

        out["F"] = f
        out["G"] = anp.column_stack(g123 + g456)
        out["H"] = anp.column_stack(h + [h13, h14])

    def _calc_pareto_set(self):
        return [
            9.53E-7,
            0,
            4.21E-3,
            1.039E-4,
            0,
            0,
            2.072E-1,
            5.979E-1,
            1.298E-1,
            3.35E-2,
            1.711E-2,
            8.827E-3,
            4.657E-10,
            0,
            0,
            0,
            0,
            0,
            2.868E-4,
            1.193E-3,
            8.332E-5,
            1.239E-4,
            2.07E-5,
            1.829E-5
        ]


class G21(G):

    def __init__(self):
        xl = np.array([0, 0, 0, 100, 6.3, 5.9, 4.5], dtype=float)
        xu = np.array([1000, 40, 40, 300, 6.7, 6.4, 6.25], dtype=float)
        super().__init__(n_var=7, n_obj=1, n_ieq_constr=1, n_eq_constr=5, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = x[:, 0]
        g = -x[:, 0] + 35 * (x[:, 1] ** (0.6)) + 35 * (x[:, 2] ** 0.6)
        h1 = -300 * x[:, 2] + 7500 * x[:, 4] - 7500 * x[:, 5] - 25 * x[:, 3] * x[:, 4] + 25 * x[:, 3] * x[:, 5] + x[:,
                                                                                                                  2] * x[
                                                                                                                       :,
                                                                                                                       3]
        h2 = 100 * x[:, 1] + 155.365 * x[:, 3] + 2500 * x[:, 6] - x[:, 1] * x[:, 3] - 25 * x[:, 3] * x[:, 6] - 15536.5
        h3 = -x[:, 4] + anp.log(-x[:, 3] + 900)
        h4 = -x[:, 5] + anp.log(x[:, 3] + 300)
        h5 = -x[:, 6] + anp.log(-2 * x[:, 3] + 700)

        out["F"] = f
        out["G"] = g
        out["H"] = anp.column_stack([h1, h2, h3, h4, h5])

    def _calc_pareto_set(self):
        return [
            193.724510070034967,
            5.56944131553368433 * (10 ** -27),
            17.3191887294084914,
            100.047897801386839,
            6.68445185362377892,
            5.99168428444264833,
            6.21451648886070451
        ]


class G22(G):

    def __init__(self):
        xl = np.array([0, 0, 0, 0, 0, 0, 0, 100, 100, 100.01, 100, 100, 0, 0, 0, 0.01, 0.01, -4.7, -4.7, -4.7,
                       -4.7, -4.7], dtype=float)

        xu = np.array([20000, 10 ** 6, 10 ** 6, 10 ** 6, 4 * (10 ** 7), 4 * (10 ** 7), 4 * (10 ** 7), 299.99, 399.99,
                       300, 400, 600, 500, 500, 500, 300, 400, 6.25, 6.25, 6.25, 6.25, 6.25], dtype=float)
        super().__init__(n_var=22, n_obj=1, n_ieq_constr=1, n_eq_constr=19, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = x[:, 0]

        g = -x[:, 0] + x[:, 1] ** 0.6 + x[:, 2] ** 0.6 + x[:, 3] ** 0.6

        h1 = x[:, 4] - 100000 * x[:, 7] + 10 ** 7
        h2 = x[:, 5] + 100000 * x[:, 7] - 100000 * x[:, 8]
        h3 = x[:, 6] + 100000 * x[:, 8] - 5 * 10 ** 7
        h4 = x[:, 4] + 100000 * x[:, 9] - 3.3 * 10 ** 7
        h5 = x[:, 5] + 100000 * x[:, 10] - 4.4 * 10 ** 7
        h6 = x[:, 6] + 100000 * x[:, 11] - 6.6 * 10 ** 7
        h7 = x[:, 4] - 120 * x[:, 1] * x[:, 12]
        h8 = x[:, 5] - 80 * x[:, 2] * x[:, 13]
        h9 = x[:, 6] - 40 * x[:, 3] * x[:, 14]
        h10 = x[:, 7] - x[:, 10] + x[:, 15]
        h11 = x[:, 8] - x[:, 11] + x[:, 16]
        h12 = -x[:, 17] + anp.log(x[:, 9] - 100)
        h13 = -x[:, 18] + anp.log(-x[:, 7] + 300)
        h14 = -x[:, 19] + anp.log(x[:, 15])
        h15 = -x[:, 20] + anp.log(-x[:, 8] + 400)
        h16 = -x[:, 21] + anp.log(x[:, 16])
        h17 = -x[:, 7] - x[:, 9] + x[:, 12] * x[:, 17] - x[:, 12] * x[:, 18] + 400
        h18 = x[:, 7] - x[:, 8] - x[:, 10] + x[:, 13] * x[:, 19] - x[:, 13] * x[:, 20] + 400
        h19 = x[:, 8] - x[:, 11] - 4.60517 * x[:, 14] + x[:, 14] * x[:, 21] + 100

        out["F"] = f
        out["G"] = g
        out["H"] = anp.column_stack(
            [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19])

    def _calc_pareto_set(self):
        return [
            236.430975504001054,
            135.82847151732463,
            204.818152544824585,
            6446.54654059436416,
            3007540.83940215595,
            4074188.65771341929,
            32918270.5028952882,
            130.075408394314167,
            170.817294970528621,
            299.924591605478554,
            399.258113423595205,
            330.817294971142758,
            184.51831230897065,
            248.64670239647424,
            127.658546694545862,
            269.182627528746707,
            160.000016724090955,
            5.29788288102680571,
            5.13529735903945728,
            5.59531526444068827,
            5.43444479314453499,
            5.07517453535834395
        ]


class G23(G):

    def __init__(self):
        xl = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.01], dtype=float)
        xu = np.array([300, 300, 100, 200, 100, 300, 100, 200, 0.03], dtype=float)
        super().__init__(n_var=9, n_obj=1, n_ieq_constr=2, n_eq_constr=4, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = -9 * x[:, 4] - 15 * x[:, 7] + 6 * x[:, 0] + 16 * x[:, 1] + 10 * (x[:, 5] + x[:, 6])

        g1 = x[:, 8] * x[:, 2] + 0.02 * x[:, 5] - 0.025 * x[:, 4]
        g2 = x[:, 8] * x[:, 3] + 0.02 * x[:, 6] - 0.015 * x[:, 7]

        h1 = x[:, 0] + x[:, 1] - x[:, 2] - x[:, 3]
        h2 = 0.03 * x[:, 0] + 0.01 * x[:, 1] - x[:, 8] * (x[:, 2] + x[:, 3])
        h3 = x[:, 2] + x[:, 5] - x[:, 4]
        h4 = x[:, 3] + x[:, 6] - x[:, 7]

        out["F"] = f
        out["G"] = anp.column_stack([g1, g2])
        out["H"] = anp.column_stack([h1, h2, h3, h4])

    def _calc_pareto_set(self):
        return [0, 100, 0, 100, 0, 0, 100, 200, 0.01]


class G24(G):

    def __init__(self):
        xl = np.array([0, 0], dtype=float)
        xu = np.array([3, 4], dtype=float)
        super().__init__(n_var=2, n_obj=1, n_ieq_constr=2, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = -x[:, 0] - x[:, 1]
        g1 = -2 * x[:, 0] ** 4 + 8 * x[:, 0] ** 3 - 8 * x[:, 0] ** 2 + x[:, 1] - 2
        g2 = -4 * x[:, 0] ** 4 + 32 * x[:, 0] ** 3 - 88 * x[:, 0] ** 2 + 96 * x[:, 0] + x[:, 1] - 36

        out["F"] = f
        out["G"] = anp.column_stack([g1, g2])

    def _calc_pareto_set(self):
        return [2.329520197477607, 3.17849307411768]
