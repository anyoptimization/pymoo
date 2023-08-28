from pymoo.util.display.column import Column
from pymoo.util.display.output import Output, pareto_front_if_possible


class MinimumConstraintViolation(Column):

    def __init__(self, **kwargs) -> None:
        super().__init__("cv_min", **kwargs)

    def update(self, algorithm):
        self.value = algorithm.opt.get("cv").min()


class AverageConstraintViolation(Column):

    def __init__(self, **kwargs) -> None:
        super().__init__("cv_avg", **kwargs)

    def update(self, algorithm):
        self.value = algorithm.pop.get("cv").mean()


class SingleObjectiveOutput(Output):

    def __init__(self):
        super().__init__()
        self.cv_min = MinimumConstraintViolation()
        self.cv_avg = AverageConstraintViolation()

        self.f_min = Column(name="f_min")
        self.f_avg = Column(name="f_avg")
        self.f_gap = Column(name="f_gap")

        self.best = None

    def initialize(self, algorithm):
        problem = algorithm.problem

        if problem.has_constraints():
            self.columns += [self.cv_min, self.cv_avg]

        self.columns += [self.f_avg, self.f_min]

        pf = pareto_front_if_possible(problem)
        if pf is not None:
            self.best = pf.flatten()[0]
            self.columns += [self.f_gap]

    def update(self, algorithm):
        super().update(algorithm)

        f, cv, feas = algorithm.pop.get("f", "cv", "feas")

        if feas.sum() > 0:
            self.f_avg.set(f[feas].mean())
        else:
            self.f_avg.set(None)

        opt = algorithm.opt[0]

        if opt.feas:
            self.f_min.set(opt.f)
            if self.best:
                self.f_gap.set(opt.f - self.best)

        else:
            self.f_min.set(None)
            self.f_gap.set(None)
