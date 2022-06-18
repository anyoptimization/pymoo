import numpy as np

from pymoo.core.repair import Repair


class RoundingRepair(Repair):

    def __init__(self, **kwargs) -> None:
        """

        Returns
        -------
        object
        """
        super().__init__(**kwargs)

    def _do(self, problem, X, **kwargs):
        return np.around(X).astype(int)
