import autograd.numpy as anp

from pymoo.model.problem import Problem


class Clutch(Problem):
    def __init__(self):
        super().__init__(n_var=5, n_obj=2, n_constr=19, type_var=anp.int)
        # ri, ro, t, F, Z
        # self.xl = anp.array([60, 90, 1, 600, 2])
        self.xl = anp.array([0, 0, 0, 0, 0])
        self.xu = anp.array([20, 20, 4, 400, 7])

        self.x1 = anp.arange(60, 81)
        self.x2 = anp.arange(90, 111)
        self.x3 = anp.arange(1, 3.5, 0.5)
        self.x4 = anp.arange(600, 1001)
        self.x5 = anp.arange(2, 11)

    def _evaluate(self, x, out, *args, **kwargs):

        x1, x2, x3, x4, x5 = anp.split(x, 5, axis=1)

        x1 = self.x1[x1]
        x2 = self.x2[x2]
        x3 = self.x3[x3]
        x4 = self.x4[x4]
        x5 = self.x5[x5]

        pi = anp.pi
        mu = 0.5
        s = 1.5
        M_f = 3  # Nm
        ri_max = 80  # mm
        t_max = 3  # mm
        n = 250  # rpm
        w = pi * n/30  # rad/s
        R_sr = (2/3) * ((x2**3 - x1**3)/(x2**2 - x1**2))  # mm
        p_max = 1  # MPa
        T_max = 15  # s
        I_z = 55  # kg*m^2
        ro_min = 90  # mm
        F_max = 1000  # N
        A = pi * (x2**2 - x1**2)  # mm^2
        deltaR = 20  # mm
        rho = 0.0000078  # kg/mm^2
        delta = 0.5  # mm
        ro_max = 110  # mm

        Z_max = 9
        p_rz = x4/A  # N/mm^2
        L_max = 30  # mm
        Vsr_max = 10  #m/s
        M_s = 40  # Nm
        ri_min = 60  # mm
        t_min = 1.5  # mm

        M_h = (2/3) * mu * x4 * x5 * ((x2**3 - x1**3)/(x2**2 - x1**2))  # N*mm
        Vsr = (pi * R_sr * n)/30  # mm/s

        T = (I_z * w)/(M_h/1000 + M_f)

        g1 = (x2 - x1 - deltaR) * -1
        g2 = (L_max - (x5 + 1) * (x3 + delta)) * -1
        g3 = (p_max - p_rz) * -1
        g4 = (p_max * Vsr_max*1000 - p_rz * Vsr) * -1
        g5 = (Vsr_max*1000 - Vsr) * -1
        g6 = (M_h/1000 - (s * M_s)) * -1
        g7 = T * -1
        g8 = (T_max - T) * -1

        _g9 = -x1 + ri_min
        _g10 = x1 - ri_max

        _g11 = -x2 + ro_min
        _g12 = x2 - ro_max

        _g13 = -x3 + t_min
        _g14 = x3 - t_max

        _g15 = -x4
        _g16 = x4 - F_max

        _g17 = -x5 + 2
        _g18 = x5 - Z_max

        f1 = pi * (x2**2 - x1**2) * x3 * (x5 + 1) * rho
        f2 = T

        out["F"] = anp.column_stack([f1, f2])
        out["G"] = anp.column_stack([g1, g2, g3, g4, g5, g6, g7, g8])


if __name__ == "__main__":
    x = anp.array([[10, 0, 1, 400, 1]])
    # x = anp.array([[10, 0, 0, 260, 1]])

    problem = Clutch()

    res = problem.evaluate(x)

    print(res)