import numpy as np

from pymoo.util.ref_dirs.optimizer import GradientDescent


class CoordinateDescent(GradientDescent):

    def direction(self, dF, **kwargs):
        dF = dF[0]
        p = np.zeros(len(dF))
        k = np.abs(dF).argmax()
        p[k] = 1 * np.sign(dF[k])
        return - p
