from pymoo.performance_indicator.distance_indicator import DistanceIndicator, modified_distance


class GDPlus(DistanceIndicator):

    def __init__(self, pf, **kwargs):
        super().__init__(pf, modified_distance, 0, **kwargs)
