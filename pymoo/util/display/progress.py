"""Progress bar wrapper for optimization visualization."""

import math
from typing import Any

from alive_progress import alive_bar


class ProgressBar:
    """Progress bar for displaying optimization progress.

    Parameters:
        *args: Positional arguments passed to alive_bar.
        start: Whether to start the progress bar immediately.
        non_decreasing: Whether to ensure progress only increases.
        **kwargs: Keyword arguments passed to alive_bar.
    """

    def __init__(self, *args: Any, start: bool = True, non_decreasing: bool = True, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

        for key, default in [("manual", True), ("force_tty", True)]:
            if key not in kwargs:
                kwargs[key] = default

        self.func: Any = None
        self.obj: Any = None
        self.non_decreasing = non_decreasing
        self._max = 0.0

        if start:
            self.start()

    def set(self, value: float, *args: Any, **kwargs: Any) -> None:
        """Set the progress bar value.

        Args:
            value: Progress value between 0 and 1.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        if self.non_decreasing:
            self._max = max(self._max, value)
            value = self._max

        prec = 100
        value = math.floor(value * prec) / prec

        self.obj(value, *args, **kwargs)

    def start(self) -> None:
        """Start the progress bar."""
        if not self.obj:
            self.func = alive_bar(*self.args, **self.kwargs).gen

            self.obj = next(self.func)

    def close(self) -> None:
        """Close the progress bar."""
        if self.obj:
            try:
                next(self.func)
            except:  # noqa: E722
                pass

    def __enter__(self) -> "ProgressBar":
        """Enter context manager."""
        self.start()
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        """Exit context manager."""
        self.close()
