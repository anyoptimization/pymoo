import numpy as np

from pymoo.constraints.tcv import TotalConstraintViolation


class Default:

    tcv = TotalConstraintViolation(ieq_eps=0.0, ieq_pow=None, eq_eps=1e-4, eq_pow=None, aggr_func=np.mean)

