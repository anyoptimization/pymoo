class Observer:

    def __init__(self) -> None:
        super().__init__()
        self.data = {}

    def notify(self, algorithm, event):
        return getattr(self, event)(algorithm)

    def setup(self, algorithm):
        pass

    def initialize(self, algorithm):
        pass

    def update(self, algorithm):
        pass

    def finalize(self, algorithm):
        pass


class Observable:

    def __init__(self):
        self.observers = []

    def register(self, observer):
        self.observers.append(observer)

    def event(self, name):
        for obs in self.observers:
            obs.notify(self, name)


