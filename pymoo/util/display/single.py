"""Single-objective optimization output display."""

from typing import Any

from pymoo.util.display.column import Column
from pymoo.util.display.output import Output, pareto_front_if_possible


class MinimumConstraintViolation(Column):
    """Column displaying minimum constraint violation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("cv_min", **kwargs)

    def update(self, algorithm: Any) -> None:
        """Update the column value with minimum constraint violation.

        Args:
            algorithm: The optimization algorithm instance.
        """
        self.value = algorithm.opt.get("cv").min()


class AverageConstraintViolation(Column):
    """Column displaying average constraint violation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("cv_avg", **kwargs)

    def update(self, algorithm: Any) -> None:
        """Update the column value with average constraint violation.

        Args:
            algorithm: The optimization algorithm instance.
        """
        self.value = algorithm.pop.get("cv").mean()


class SingleObjectiveOutput(Output):
    """Output columns for single-objective optimization results."""

    def __init__(self) -> None:
        super().__init__()
        self.cv_min = MinimumConstraintViolation()
        self.cv_avg = AverageConstraintViolation()

        self.f_min = Column(name="f_min")
        self.f_avg = Column(name="f_avg")
        self.f_gap = Column(name="f_gap")

        self.best: Any = None

    def initialize(self, algorithm: Any) -> None:
        """Initialize output columns based on the problem.

        Args:
            algorithm: The optimization algorithm instance.
        """
        problem = algorithm.problem

        if problem.has_constraints():
            self.columns += [self.cv_min, self.cv_avg]

        self.columns += [self.f_avg, self.f_min]

        pf = pareto_front_if_possible(problem)
        if pf is not None:
            self.best = pf.flatten()[0]
            self.columns += [self.f_gap]

    def update(self, algorithm: Any) -> None:
        """Update output columns with current algorithm state.

        Args:
            algorithm: The optimization algorithm instance.
        """
        super().update(algorithm)

        f, cv, feas = algorithm.pop.get("f", "cv", "feas")

        if feas.sum() > 0:
            self.f_avg.set(f[feas].mean())
        else:
            self.f_avg.set(None)

        opt = algorithm.opt[0]

        if opt.feas:
            self.f_min.set(opt.f)
            if self.best is not None:
                self.f_gap.set(opt.f - self.best)
        else:
            self.f_min.set(None)
            self.f_gap.set(None)
