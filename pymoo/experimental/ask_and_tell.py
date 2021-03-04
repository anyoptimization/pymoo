from pymoo.model.evaluator import Evaluator


class AskAndTell:

    def __init__(self, algorithm) -> None:
        super().__init__()
        algorithm.evaluator = Evaluator()
        self.algorithm = algorithm
        self.problem = None

    def has_next(self):
        return self.algorithm.has_next()

    def ask(self):
        algo = self.algorithm

        if algo.is_initialized:
            return algo.infill()
        else:
            algo._initialize()
            return algo._initialize_infill()

    def tell(self, pop):
        algo = self.algorithm

        if not algo.is_initialized:
            algo.pop, algo.off = pop, pop
            algo._post_initialize()
            algo._set_optimum()
            algo.is_initialized = True
        else:
            algo.advance(infills=pop)
