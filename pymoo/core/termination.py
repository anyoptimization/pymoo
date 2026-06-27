"""Termination criteria for optimization algorithms."""

from abc import abstractmethod


class Termination:
    """Base class for termination criteria."""

    def __init__(self) -> None:
        """Initialize termination criterion."""
        super().__init__()
        self.force_termination = False
        self.perc = 0.0

    def update(self, algorithm: object) -> float:
        """Update progress and check termination condition.

        Args:
            algorithm: Algorithm instance.

        Returns:
            Progress value in [0, 1].
        """
        if self.force_termination:
            progress = 1.0
        else:
            progress = self._update(algorithm)
            assert progress >= 0.0, (
                "Invalid progress was set by the TerminationCriterion."
            )

        self.perc = progress
        return self.perc

    def has_terminated(self) -> bool:
        """Check whether termination criterion is satisfied."""
        return self.perc >= 1.0

    def do_continue(self) -> bool:
        """Check whether optimization should continue."""
        return not self.has_terminated()

    def terminate(self) -> None:
        """Force termination."""
        self.force_termination = True

    @abstractmethod
    def _update(self, algorithm: object) -> float:
        """Update progress (to be implemented by subclasses)."""


class NoTermination(Termination):
    """Termination criterion that never terminates."""

    def _update(self, algorithm: object) -> float:
        """Never terminate."""
        return 0.0


class MultipleCriteria(Termination):
    """Base class for combining multiple termination criteria."""

    def __init__(self, *args: Termination) -> None:
        """Initialize with multiple criteria.

        Args:
            args: Termination criteria to combine.
        """
        super().__init__()
        self.criteria = args


class TerminateIfAny(MultipleCriteria):
    """Terminate if any criterion is satisfied."""

    def _update(self, algorithm: object) -> float:
        """Terminate if any criterion is met."""
        return max([termination.update(algorithm) for termination in self.criteria])


class TerminateIfAll(MultipleCriteria):
    """Terminate only if all criteria are satisfied."""

    def _update(self, algorithm: object) -> float:
        """Terminate only if all criteria are met."""
        return min([termination.update(algorithm) for termination in self.criteria])
