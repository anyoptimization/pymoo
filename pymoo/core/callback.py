class Callback:

    def __init__(self) -> None:
        super().__init__()
        self.data = {}

    def notify(self, algorithm, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.notify(*args, **kwargs)
