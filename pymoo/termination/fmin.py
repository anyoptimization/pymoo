from pymoo.core.termination import Termination


class MinimumFunctionValueTermination(Termination):

    def __init__(self, fmin) -> None:
        super().__init__()
        self.fmin = fmin

    def _update(self, algorithm):
        opt = algorithm.opt

        if not any(opt.get("feas")):
            return 0.0
        else:
            return self.fmin / opt.get("F").min()
