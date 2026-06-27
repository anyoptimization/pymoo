"""NSDER - Many-objective differential evolution with reference directions."""

import numpy as np

from pymoo.algorithms.moo.nsde import NSDE
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.util.misc import has_feasible


class NSDER(NSDE):
    """Many-objective DE with reference-direction survival (NSDE-R).

    Extends NSDE to many-objective problems using NSGA-III reference-direction survival.

    Reference: S. R. Reddy & G. S. Dulikravich, "Many-objective differential evolution
    optimization based on reference points: NSDE-R," Struct. Multidisc. Optim., 2019.
    """

    def __init__(
        self,
        ref_dirs,
        pop_size=None,
        variant="DE/rand/1/bin",
        CR=0.7,
        F=None,
        gamma=1e-4,
        **kwargs,
    ):
        """Initialize NSDER algorithm.

        Args:
            ref_dirs: Reference directions (shape: [n_dirs, n_obj]).
            pop_size: Population size. Defaults to len(ref_dirs).
            variant: DE strategy string. Defaults to 'DE/rand/1/bin'.
            CR: Crossover rate. Defaults to 0.7.
            F: Scale factor(s). Defaults to randomized in (0, 1).
            gamma: Jitter deviation. Defaults to 1e-4.
            **kwargs: Additional arguments (including de_repair for donor vector repair).
        """
        self.ref_dirs = ref_dirs

        if pop_size is None:
            pop_size = len(ref_dirs)

        if len(ref_dirs) > pop_size:
            print(
                f"WARNING: pop_size={pop_size} is less than the number of reference directions "
                f"ref_dirs={len(ref_dirs)}.\n"
                "This might cause unwanted behavior. "
                "Please make sure pop_size >= number of reference directions."
            )

        if "survival" in kwargs:
            survival = kwargs.pop("survival")
        else:
            survival = ReferenceDirectionSurvival(ref_dirs)

        super().__init__(
            pop_size=pop_size,
            variant=variant,
            CR=CR,
            F=F,
            gamma=gamma,
            survival=survival,
            **kwargs,
        )

    def _setup(self, problem, **kwargs):
        if self.ref_dirs is not None and self.ref_dirs.shape[1] != problem.n_obj:
            raise ValueError(
                "Dimensionality of reference directions must equal the number of objectives: "
                f"{self.ref_dirs.shape[1]} != {problem.n_obj}"
            )

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.survival.opt
