from pymoo.indicators.distance_indicator import DistanceIndicator, euclidean_distance


class IGD(DistanceIndicator):

    def __init__(self, pf, **kwargs):
        super().__init__(pf, euclidean_distance, 1, **kwargs)
