import numpy as np

from pymoo.problems.dyn import DynamicTestProblem


class DynamicCEC2015(DynamicTestProblem):

    def __init__(self, n_var=10, nt=10, taut=20, n_obj=2, xl=0.0, xu=1.0, vtype=float, **kwargs):
        super().__init__(nt, taut, n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=vtype, **kwargs)


class FDA2DEB(DynamicCEC2015):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import fda2_deb as f
        out["F"] = np.array([f(x, t) for x in X])


class FDA4(DynamicCEC2015):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, n_obj=3, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import FDA4 as f
        out["F"] = np.array([f(x, t) for x in X])


class FDA5(DynamicCEC2015):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, n_obj=3, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import FDA5 as f
        out["F"] = np.array([f(x, t) for x in X])


class DIMP2(DynamicCEC2015):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import DIMP2 as f
        out["F"] = np.array([f(x, t) for x in X])


class dMOP2(DynamicCEC2015):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import dMOP2 as f
        out["F"] = np.array([f(x, t) for x in X])


class dMOP3(DynamicCEC2015):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import dMOP3 as f
        out["F"] = np.array([f(x, t) for x in X])


class HE2(DynamicCEC2015):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import HE2 as f
        out["F"] = np.array([f(x, t) for x in X])


class HE7(DynamicCEC2015):

    def __init__(self, n_var=10, **kwargs):
        super().__init__(n_var=n_var, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import HE7 as f
        out["F"] = np.array([f(x, t) for x in X])


class HE9(DynamicCEC2015):

    def __init__(self, n_var=10, **kwargs):
        super().__init__(n_var=n_var, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import HE9 as f
        out["F"] = np.array([f(x, t) for x in X])
