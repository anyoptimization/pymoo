"""Lazy backend module that provides transparent access to the active gradient backend."""

import sys
import importlib
from typing import Any


class LazyBackend:
    """Lazy backend that always uses the current active backend."""

    def __getattr__(self, name: str) -> Any:
        from pymoo.gradient import active_backend, BACKENDS

        backend_module = importlib.import_module(BACKENDS[active_backend])
        return getattr(backend_module, name)

    def __dir__(self) -> list[str]:
        """Support for tab completion and introspection."""
        from pymoo.gradient import active_backend, BACKENDS

        backend_module = importlib.import_module(BACKENDS[active_backend])
        return dir(backend_module)


# Replace this module with the lazy backend
sys.modules[__name__] = LazyBackend()  # type: ignore[assignment]
