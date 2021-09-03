from pymoo.indicators.distance_indicator import DistanceIndicator, euclidean_distance


class GD(DistanceIndicator):

    def __init__(self, pf, **kwargs):
        super().__init__(pf, euclidean_distance, 0, **kwargs)
