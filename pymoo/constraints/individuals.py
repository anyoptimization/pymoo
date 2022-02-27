from pymoo.core.individual import Individual


class ConstrainedIndividual(Individual):

    def __init__(self, tcv) -> None:
        self.tcv = tcv

    @property
    def CV(self):
        return self.tcv.calc(self.G, self.H)

    @property
    def FEAS(self):
        return self.CV <= self.tcv.feas_eps

    def copy(self, **kwargs):
        obj = self.__class__(self.tcv)
        for k, v in kwargs.items():
            obj.__dict__[k] = v
        return obj


