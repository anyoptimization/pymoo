class Cache:

    def __init__(self, func, raise_exception=True) -> None:
        super().__init__()

        # the function to be executed
        self.func = func

        # the result object which is cached
        self.cache = None

        # a flag indicating if it has been executed
        self.has_been_executed = False

        # whether an exception shall be thrown if it fails
        self.raise_exception = raise_exception

        # the error message if an exception occurs
        self.error = None

    def exec(self, *args, use_cache=True, **kwargs):

        if use_cache and self.has_been_executed:
            return self.cache

        else:
            try:
                self.cache = self.func(*args, **kwargs)
            except Exception as e:
                self.error = str(e)

                if self.raise_exception:
                    raise e

            self.has_been_executed = True

            return self.cache

    def reset(self):
        self.has_been_executed = False
        self.cache = None
        self.error = None
