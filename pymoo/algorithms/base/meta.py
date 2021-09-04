from pymoo.core.algorithm import Algorithm


class MetaAlgorithm(Algorithm):

    def __init__(self,
                 clazz,
                 *args,
                 display=None,
                 **kwargs
                 ):
        super().__init__(*args, display=display, **kwargs)

        self.clazz = clazz
        self.args = args
        self.kwargs = kwargs
        self.algorithm = self.create()

    def create(self, *args, **kwargs):
        return self.clazz(*self.args, *args, **self.kwargs, **kwargs)

    def setup(self, *args, **kwargs):
        self.algorithm.setup(*args, **kwargs)
        super().setup(*args, **kwargs)

    def _initialize_infill(self):
        return self._infill()

    def _initialize_advance(self, **kwargs):
        return self._advance(**kwargs)

    def _infill(self):
        return self.algorithm.infill()

    def _advance(self, infills=None, **kwargs):
        return self.algorithm.advance(infills=infills, **kwargs)

    def _set_optimum(self):
        self.opt = self.algorithm.opt



