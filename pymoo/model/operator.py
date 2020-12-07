from pymoo.util.type_converter import DefaultConverter


class Operator:

    def __init__(self, converter=DefaultConverter()) -> None:
        super().__init__()
        self.converter = converter