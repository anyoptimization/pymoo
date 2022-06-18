from pymoo.indicators.gd import GD
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.termination.ftol import MultiObjectiveSpaceTermination

from pymoo.util.display.column import Column
from pymoo.util.display.output import Output, pareto_front_if_possible
from pymoo.util.display.single import MinimumConstraintViolation, AverageConstraintViolation


class NumberOfNondominatedSolutions(Column):

    def __init__(self, width=6, **kwargs) -> None:
        super().__init__("n_nds", width=width, **kwargs)

    def update(self, algorithm):
        self.value = len(algorithm.opt)


class MultiObjectiveOutput(Output):

    def __init__(self):
        super().__init__()
        self.cv_min = MinimumConstraintViolation()
        self.cv_avg = AverageConstraintViolation()
        self.n_nds = NumberOfNondominatedSolutions()

        self.igd = Column("igd")
        self.gd = Column("gd")
        self.hv = Column("hv")
        self.eps = Column("eps")
        self.indicator = Column("indicator")

        self.pf = None
        self.indicator_no_pf = None

    def initialize(self, algorithm):
        problem = algorithm.problem

        self.columns += [self.n_nds]

        if problem.has_constraints():
            self.columns += [self.cv_min, self.cv_avg]

        self.pf = pareto_front_if_possible(problem)
        if self.pf is not None:
            self.columns += [self.igd, self.gd]

            if problem.n_obj == 2:
                self.columns += [self.hv]

        else:
            self.indicator_no_pf = MultiObjectiveSpaceTermination()
            self.columns += [self.eps, self.indicator]

    def update(self, algorithm):
        super().update(algorithm)

        for col in [self.igd, self.gd, self.hv, self.eps, self.indicator]:
            col.set(None)

        F, feas = algorithm.opt.get("F", "feas")
        F = F[feas]

        if len(F) > 0:

            if self.pf is not None:

                if feas.sum() > 0:
                    self.igd.set(IGD(self.pf, zero_to_one=True).do(F))
                    self.gd.set(GD(self.pf, zero_to_one=True).do(F))

                    if self.hv in self.columns:
                        self.hv.set(Hypervolume(pf=self.pf, zero_to_one=True).do(F))

            if self.indicator_no_pf is not None:

                ind = self.indicator_no_pf
                ind.update(algorithm)

                valid = ind.delta_ideal is not None

                if valid:

                    if ind.delta_ideal > ind.tol:
                        max_from = "ideal"
                        eps = ind.delta_ideal
                    elif ind.delta_nadir > ind.tol:
                        max_from = "nadir"
                        eps = ind.delta_nadir
                    else:
                        max_from = "f"
                        eps = ind.delta_f

                    self.eps.set(eps)
                    self.indicator.set(max_from)
