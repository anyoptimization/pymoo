class Callback:

    def __init__(self) -> None:
        super().__init__()
        self.data = {}

    def notify(self, algorithm, **kwargs):
        pass
