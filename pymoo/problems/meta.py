from pymoo.model.problem import Problem


class MetaProblem(Problem):

    def __init__(self, problem, override_evaluate=False):
        self.__dict__.update(**problem.__dict__)
        self.problem = problem
        self.override_evaluate = override_evaluate

    def do(self, X, out, *args, **kwargs):
        obj = super() if not self.override_evaluate else self.problem
        obj.do(X, out, *args, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        self.problem._evaluate(x, out, *args, **kwargs)

    def pareto_front(self, *args, **kwargs):
        return self.problem.pareto_front(*args, **kwargs)

    def pareto_set(self, *args, **kwargs):
        return self.problem.pareto_set(*args, **kwargs)

