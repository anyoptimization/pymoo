"""Utility functions and decorators for pymoo."""

from typing import Any, Callable, Optional, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])


def default_random_state(func_or_seed: Optional[Any] = None, *, seed: Optional[int] = None) -> Any:
    """Decorator that provides a default random state to functions.

    Can be used as:
    - @default_random_state
    - @default_random_state(1)  # with positional seed
    - @default_random_state(seed=1)  # with keyword seed

    If random_state is provided to the function call, it takes precedence.

    Args:
        func_or_seed: Function to decorate or seed value.
        seed: Default seed for random state generation.

    Returns:
        Decorated function or decorator callable.
    """

    def decorator(func: F, default_seed: Optional[int] = None) -> Callable[..., Any]:
        def wrapper(*args: Any, random_state: Any = None, **kwargs: Any) -> Any:
            if random_state is None:
                seed_to_use = kwargs.pop("seed", default_seed)
                random_state = np.random.default_rng(seed_to_use)
            return func(*args, random_state=random_state, **kwargs)

        return wrapper

    if func_or_seed is None:
        return lambda func: decorator(func, seed)
    elif callable(func_or_seed):
        return decorator(func_or_seed, None)
    else:
        return lambda func: decorator(func, func_or_seed)
