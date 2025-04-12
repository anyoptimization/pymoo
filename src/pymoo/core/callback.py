class Callback:

    def __init__(self) -> None:
        super().__init__()
        self.data = {}
        self.is_initialized = False

    def initialize(self, algorithm):
        pass

    def notify(self, algorithm):
        pass

    def update(self, algorithm):
        return self._update(algorithm)

    def _update(self, algorithm):
        pass

    def __call__(self, algorithm):

        if not self.is_initialized:
            self.initialize(algorithm)
            self.is_initialized = True

        self.notify(algorithm)
        self.update(algorithm)


class CallbackCollection(Callback):

    def __init__(self, *args) -> None:
        super().__init__()
        self.callbacks = args

    def update(self, algorithm):
        [callback.update(algorithm) for callback in self.callbacks]

