import typing
from typing import Any, Optional

import numpy as np

if typing.TYPE_CHECKING:
    from pymoo.core.population import Population
    from pymoo.core.problem import Problem
    from pymoo.util.archive import Archive
    from pymoo.core.algorithm import Algorithm


class Result:
    """
    The resulting object of an optimization run.
    """

    def __init__(self) -> None:
        super().__init__()

        self.opt: Population | None = None
        self.success = None
        self.message = None

        # ! other attributes to be set as well

        # the problem that was solved
        self.problem: Optional[Problem] = None

        # the archive stored during the run
        self.archive: Optional[Archive] = None

        # the optimal solution for that problem
        self.pf = None

        # the algorithm that was used for optimization
        self.algorithm: Optional[Algorithm] = None

        # the final population if it applies
        self.pop: Optional[Population] = None

        # directly the values of opt
        self.X: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None
        self.CV: Optional[np.ndarray] = None
        self.G: Optional[np.ndarray] = None
        self.H: Optional[np.ndarray] = None

        # all the timings that are stored of the run
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.exec_time: Optional[float] = None

        # the history of the optimization run is they were saved
        self.history: list[Algorithm] = []

        # data stored within the algorithm
        self.data: dict[Any, Any] | None = None

    @property
    def cv(self):
        return self.CV[0]

    @property
    def f(self):
        return self.F[0]

    @property
    def feas(self):
        return self.cv <= 0
