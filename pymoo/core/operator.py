import abc


class Operator:

    def __init__(self, name=None) -> None:
        super().__init__()

        if name is None:
            name = self.__class__.__name__

        self.name = name

    @abc.abstractmethod
    def do(self, *args, **kwargs):
        pass


