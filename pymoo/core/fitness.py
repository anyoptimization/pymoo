import numpy as np

from pymoo.core.survival import Survival


# =========================================================================================================
# Implementation
# =========================================================================================================

class BaseFitnessSurvival(Survival):
    
    def __init__(self) -> None:
        super().__init__(filter_infeasible=False)
    
    def _do(self, problem, pop, n_survive=None, **kwargs):
        return pop[:n_survive]
    

class FitnessSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=False)

    def _do(self, problem, pop, n_survive=None, **kwargs):
        F, cv = pop.get("F", "cv")
        assert F.shape[1] == 1, "FitnessSurvival can only used for single objective single!"
        S = np.lexsort([F[:, 0], cv])
        pop.set("rank", np.argsort(S))
        return pop[S[:n_survive]]
