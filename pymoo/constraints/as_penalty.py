import numpy as np

from pymoo.core.individual import calc_cv
from pymoo.core.problem import MetaProblem
from pymoo.util.misc import from_dict


class ConstraintsAsPenalty(MetaProblem):

    def __init__(self,
                 problem,
                 penalty: float = 0.1):
        # Initialize MetaProblem with the wrapped problem
        super().__init__(problem)

        # the amount of penalty to add for this type
        self.penalty = penalty

        # set ieq and eq to zero (because it became now a penalty)
        self.n_ieq_constr = 0
        self.n_eq_constr = 0



    def _evaluate(self, X, out, *args, **kwargs):
        """
        Implement _evaluate by calling the wrapped problem and processing constraints.
        """
        # Call the wrapped problem's evaluate method to get properly formatted output
        wrapped_out = self.__wrapped__.evaluate(X, return_as_dictionary=True, *args, **kwargs)
        
        # get at the values from the wrapped output
        F, G, H = from_dict(wrapped_out, "F", "G", "H")

        # store a backup of the values in out
        out["__F__"], out["__G__"], out["__H__"] = F, G, H

        # calculate the total constraint violation (here normalization shall be already included)
        CV = calc_cv(G=G, H=H)
        out["__CV__"] = CV

        # set the penalized objective values
        out["F"] = F + self.penalty * np.reshape(CV, F.shape)

