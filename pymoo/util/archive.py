from pymoo.core.population import Population
from pymoo.util.optimum import filter_optimum


class SingleObjectiveArchive:

    def __init__(self) -> None:
        super().__init__()
        self.sols = Population()

    def add(self, sols):
        if len(sols) == 0:
            return
        else:
            self.sols = filter_optimum(Population.merge(self.sols, sols), least_infeasible=True)
