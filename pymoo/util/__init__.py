import numpy as np


def default_random_state(func_or_seed=None, *, seed=None):
    """
    Decorator that provides a default random state to functions.
    
    Can be used as:
    - @default_random_state
    - @default_random_state(1)  # with positional seed
    - @default_random_state(seed=1)  # with keyword seed
    
    If random_state is provided to the function call, it takes precedence.
    """
    def decorator(func, default_seed=None):
        def wrapper(*args, random_state=None, **kwargs):
            if random_state is None:
                # Check if seed is provided in kwargs, otherwise use default_seed
                seed_to_use = kwargs.pop('seed', default_seed)
                random_state = np.random.default_rng(seed_to_use)
            return func(*args, random_state=random_state, **kwargs)
        return wrapper
    
    # Handle different calling patterns
    if func_or_seed is None:
        # Called as @default_random_state() or @default_random_state(seed=1)
        return lambda func: decorator(func, seed)
    elif callable(func_or_seed):
        # Called as @default_random_state (no parentheses)
        return decorator(func_or_seed, None)
    else:
        # Called as @default_random_state(1) (positional seed)
        return lambda func: decorator(func, func_or_seed)