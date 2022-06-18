from pymoo.core.repair import Repair


class TypeRepair(Repair):

    def __init__(self, vtype) -> None:
        super().__init__()
        self.vtype = vtype

    def _do(self, problem, X, **kwargs):
        return X.astype(self.vtype)
