"""Sliding window data structure for maintaining a fixed-size buffer."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


class SlidingWindow(list):  # type: ignore[type-arg]
    """Fixed-size sliding window implemented as a list."""

    def __init__(self, size: int | None = None) -> None:
        """Initialize the sliding window.

        Args:
            size: Maximum size of the window (None for unlimited).
        """
        super().__init__()
        self.size = size

    def append(self, entry: Any) -> None:
        """Append an entry and maintain window size.

        Args:
            entry: The entry to append.
        """
        super().append(entry)

        if self.size is not None:
            while len(self) > self.size:
                self.pop(0)

    def is_full(self) -> bool:
        """Check if the window is at full capacity.

        Returns:
            True if size is set and window is full, False otherwise.
        """
        return self.size == len(self)

    def to_numpy(self) -> NDArray:
        """Convert window contents to numpy array.

        Returns:
            Numpy array containing window entries.
        """
        return np.array(self)

    def clear(self) -> None:
        """Clear all entries from the window."""
        while len(self) > 0:
            self.pop()
