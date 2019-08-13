from pymoo.performance_indicator.distance_indicator import DistanceIndicator, euclidean_distance, modified_distance


class IGDPlus(DistanceIndicator):

    def __init__(self, pf, **kwargs):
        super().__init__(pf, modified_distance, 1, **kwargs)
