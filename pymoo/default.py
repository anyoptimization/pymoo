import numpy as np

from pymoo.constraints.tcv import TotalConstraintViolation


class Default:

    tcv = TotalConstraintViolation(ieq_beta=1.0, ieq_eps=0.0, eq_beta=1.0, eq_eps=1e-5, aggr_func=np.mean)

