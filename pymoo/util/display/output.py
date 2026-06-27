"""Output columns for displaying optimization progress."""

from typing import Any

from pymoo.core.callback import Callback
from pymoo.util.display.column import Column


def pareto_front_if_possible(problem: Any) -> Any:
    """Get Pareto front from problem if available.

    Args:
        problem: The optimization problem.

    Returns:
        The Pareto front array if available, None otherwise.
    """
    try:
        return problem.pareto_front()
    except:  # noqa: E722
        return None


class NumberOfGenerations(Column):
    """Column displaying the current generation number."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("n_gen", **kwargs)

    def update(self, algorithm: Any) -> None:
        """Update the column value with current generation number.

        Args:
            algorithm: The optimization algorithm instance.
        """
        self.value = algorithm.n_gen


class NumberOfEvaluations(Column):
    """Column displaying the total number of evaluations."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("n_eval", **kwargs)

    def update(self, algorithm: Any) -> None:
        """Update the column value with current evaluation count.

        Args:
            algorithm: The optimization algorithm instance.
        """
        self.value = algorithm.evaluator.n_eval


class Output(Callback):
    """Base class for displaying optimization output columns."""

    def __init__(self) -> None:
        super().__init__()
        self.n_gen = NumberOfGenerations(width=6)
        self.n_eval = NumberOfEvaluations(width=8)
        self.columns = [self.n_gen, self.n_eval]

    def update(self, algorithm: Any) -> None:
        """Update all output columns.

        Args:
            algorithm: The optimization algorithm instance.
        """
        [col.update(algorithm) for col in self.columns]

    def header(self, border: bool = False) -> str:
        """Generate header line for output columns.

        Args:
            border: Whether to add border lines.

        Returns:
            Formatted header string.
        """
        regex = " | ".join(["{}"] * len(self.columns))
        header = regex.format(*[col.name.center(col.width) for col in self.columns])

        if border:
            line = "=" * len(header)
            header = line + "\n" + header + "\n" + line

        return header

    def text(self) -> str:
        """Generate text line with column values.

        Returns:
            Formatted text string with current column values.
        """
        regex = " | ".join(["{}"] * len(self.columns))
        return regex.format(*[col.text() for col in self.columns])
