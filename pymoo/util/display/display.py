"""Display callback for optimization progress."""

from typing import Any

from pymoo.core.callback import Callback
from pymoo.util.display.progress import ProgressBar


class Display(Callback):
    """Callback to display optimization progress."""

    def __init__(self, output: Any = None, progress: bool = False, verbose: bool = False) -> None:
        super().__init__()
        self.output = output
        self.verbose = verbose
        self.progress = ProgressBar() if progress else None

    def update(self, algorithm: Any, **kwargs: Any) -> None:
        """Update the display with current algorithm state.

        Args:
            algorithm: The optimization algorithm instance.
            **kwargs: Additional keyword arguments.
        """
        output, progress = self.output, self.progress

        if self.verbose and output:
            text = ""
            header = not output.is_initialized
            output(algorithm)

            if header:
                text += output.header(border=True) + "\n"
            text += output.text()

            print(text)

        if progress:
            perc = algorithm.termination.perc
            progress.set(perc)

    def finalize(self) -> None:
        """Close the progress bar if active."""
        if self.progress:
            self.progress.close()
