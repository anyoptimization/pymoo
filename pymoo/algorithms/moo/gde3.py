"""GDE3 multi-objective differential evolution algorithm and its crowding-metric variants."""

from typing import Optional, Tuple, Union

from pymoo.algorithms.moo.nsde import NSDE
from pymoo.core.population import Population
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.dominator import get_relation


# =========================================================================================================
# GDE3
# =========================================================================================================


class GDE3(NSDE):
    def __init__(
        self,
        pop_size: int = 100,
        variant: str = "DE/rand/1/bin",
        CR: float = 0.5,
        F: Optional[Union[float, Tuple[float, float]]] = None,
        gamma: float = 1e-4,
        **kwargs,
    ):
        """GDE3 (Generalized Differential Evolution 3) extends DE to multi-objective problems.

        Each trial vector competes one-to-one with its parent: if the trial dominates, it
        replaces the parent; if the parent dominates, it is kept; otherwise both enter a
        combined pool that is trimmed to pop_size via the survival operator.

        Derived classes GDE3MNN, GDE32NN, GDE3PCD use alternative crowding metrics.

        Reference: Kukkonen & Lampinen (2005). GDE3: The third evolution step of generalized
        differential evolution. IEEE CEC 2005.

        Args:
            pop_size: Population size. Defaults to 100.
            variant: DE strategy: "DE/selection/n/crossover". Defaults to 'DE/rand/1/bin'.
            CR: Crossover rate in [0, 1]. Defaults to 0.5.
            F: Scale factor(s). Defaults to randomized in (0, 1).
            gamma: Jitter deviation. Defaults to 1e-4.
            de_repair: Donor vector repair. Defaults to 'bounce-back'.
            survival: Survival operator applied when pool > pop_size. Defaults to RankAndCrowding().
            **kwargs: Additional keyword arguments forwarded to the NSDE base class.
        """
        super().__init__(pop_size=pop_size, variant=variant, CR=CR, F=F, gamma=gamma, **kwargs)

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithm uses the AskAndTell interface; 'infills' must be provided."

        survivors = []
        for k in range(len(self.pop)):
            off, parent = infills[k], self.pop[k]
            rel = get_relation(parent, off)
            if rel == 0:
                survivors.extend([parent, off])
            elif rel == -1:
                survivors.append(off)
            else:
                survivors.append(parent)

        survivors = Population.create(*survivors)
        self.pop = self.survival.do(
            self.problem,
            survivors,
            n_survive=self.n_offsprings,
            random_state=self.random_state,
        )


class GDE3MNN(GDE3):
    """GDE3 with MNN crowding metric — recommended for many-objective problems."""

    def __init__(
        self,
        pop_size=100,
        variant="DE/rand/1/bin",
        CR=0.5,
        F=None,
        gamma=1e-4,
        **kwargs,
    ):
        super().__init__(
            pop_size,
            variant,
            CR,
            F,
            gamma,
            survival=RankAndCrowding(crowding_func="mnn"),
            **kwargs,
        )


class GDE32NN(GDE3):
    """GDE3 with 2NN crowding metric — recommended for many-objective problems."""

    def __init__(
        self,
        pop_size=100,
        variant="DE/rand/1/bin",
        CR=0.5,
        F=None,
        gamma=1e-4,
        **kwargs,
    ):
        super().__init__(
            pop_size,
            variant,
            CR,
            F,
            gamma,
            survival=RankAndCrowding(crowding_func="2nn"),
            **kwargs,
        )


class GDE3PCD(GDE3):
    """GDE3 with PCD crowding metric — recommended for bi-objective problems."""

    def __init__(
        self,
        pop_size=100,
        variant="DE/rand/1/bin",
        CR=0.5,
        F=None,
        gamma=1e-4,
        **kwargs,
    ):
        super().__init__(
            pop_size,
            variant,
            CR,
            F,
            gamma,
            survival=RankAndCrowding(crowding_func="pcd"),
            **kwargs,
        )
