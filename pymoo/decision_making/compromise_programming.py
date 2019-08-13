from pymoo.model.decision_making import DecisionMaking
from pymoo.util.normalization import normalize


class CompromiseProgramming(DecisionMaking):

    def __init__(self, metric="euclidean", **kwargs) -> None:
        super().__init__(**kwargs)
        self.metric = metric

    def _do(self, F, **kwargs):

        F, _, ideal_point, nadir_point = normalize(F,
                                                   x_min=self.ideal_point,
                                                   x_max=self.nadir_point,
                                                   estimate_bounds_if_none=True,
                                                   return_bounds=True)

        return None
