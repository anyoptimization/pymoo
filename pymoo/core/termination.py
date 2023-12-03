from pymoo.core.observer import Observer
from pymoo.core.problem import Problem


class Termination(Observer):

    def __init__(self):
        super().__init__()
        self.algorithm = None
        self.message = None
        self.status = 0.0

    def has_terminated(self):
        return self.status < 1.0

    def terminate(self, message: str = None):
        self.status = 1.0
        self.message = message


class TerminationSet(Termination):

    def __init__(self, criteria=(), op=max):
        super().__init__()
        self.criteria = criteria
        self.op = op

    def setup(self, problem: Problem):
        super().setup(problem)
        [condition.setup(problem) for condition in self.criteria]

    def update(self, algorithm: 'Algorithm'):
        for condition in self.criteria:
            condition.update(algorithm)
            self.status = self.op(self.status, condition.status)


class TerminateIfAny(TerminationSet):

    def __init__(self, criteria=()):
        super().__init__(criteria, op=max)


class TerminateIfAll(TerminationSet):

    def __init__(self, criteria=()):
        super().__init__(criteria, op=min)


class MaximumIterationTermination(Termination):

    def __init__(self, max_iter: int):
        super().__init__()
        self.max_iter = max_iter

    def update(self, algorithm: 'Algorithm'):
        self.status = algorithm.iter / self.max_iter


class MaximumEvaluationTermination(Termination):

    def __init__(self, max_fevals: int):
        super().__init__()
        self.max_fevals = max_fevals

    def update(self, algorithm: 'Algorithm'):
        self.status = algorithm.evaluator.fevals / self.max_fevals
