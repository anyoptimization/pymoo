"""Simple cache decorator for class methods."""


def Cache(func):
    """Decorator that caches the first function call result.

    This is a function decorator for class attributes. It remembers the result
    of the FIRST function call and returns it from there on. Other caches like
    LRU are difficult to use because the input can be unhashable or contain
    larger numpy arrays. Thus the user has to choose how to use this cache.
    """
    func_name = func.__name__

    def wrapper(self, *args, use_cache=True, set_cache=True, **kwargs):

        if not hasattr(self, "cache"):
            setattr(self, "cache", {})

        cache = getattr(self, "cache")

        if use_cache and func_name in cache:
            return cache[func_name]
        else:
            obj = func(self, *args, **kwargs)

            if set_cache:
                cache[func_name] = obj

            return obj

    return wrapper
