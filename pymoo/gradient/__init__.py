"""Gradient computation tools and automatic differentiation backends."""

__all__ = ["activate", "deactivate"]

BACKENDS = {
    "numpy": "numpy",
    "autograd": "autograd.numpy",
    "jax": "jax.numpy",
}

active_backend = "numpy"


def activate(name: str) -> None:
    """Activate a gradient computation backend.

    Args:
        name: Backend name ('numpy', 'autograd', or 'jax').

    Raises:
        ValueError: If backend name is unknown.
    """
    global active_backend
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Choose from: {list(BACKENDS.keys())}"
        )
    active_backend = name


def deactivate() -> None:
    """Deactivate gradient computation, reverting to numpy backend."""
    activate("numpy")
