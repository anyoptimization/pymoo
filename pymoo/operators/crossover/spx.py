from pymoo.operators.crossover.pntx import PointCrossover


class SPX(PointCrossover):

    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)
