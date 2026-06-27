"""Callback interface for algorithm execution hooks."""

from typing import Any
import typing

if typing.TYPE_CHECKING:
    from pymoo.core.algorithm import Algorithm


class Callback:
    def __init__(self) -> None:
        super().__init__()
        self.data: Any = {}
        self.is_initialized: bool = False

    def initialize(self, algorithm: "Algorithm") -> None:
        pass

    def notify(self, algorithm: "Algorithm") -> None:
        pass

    def update(self, algorithm: "Algorithm") -> Any:
        return self._update(algorithm)

    def _update(self, algorithm: "Algorithm") -> Any:
        return None

    def __call__(self, algorithm: "Algorithm"):

        if not self.is_initialized:
            self.initialize(algorithm)
            self.is_initialized = True

        self.notify(algorithm)
        self.update(algorithm)


class CallbackCollection(Callback):
    def __init__(self, *args: Callback) -> None:
        super().__init__()
        self.callbacks: tuple[Callback, ...] = args

    def update(self, algorithm: "Algorithm") -> None:
        [callback.update(algorithm) for callback in self.callbacks]
