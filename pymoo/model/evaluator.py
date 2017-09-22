
class Evaluator:

    def __init__(self, n_eval):
        self.n_eval = n_eval
        self.data = []
        self.counter = 0

    def eval(self, problem, x):
        self.counter += 1
        return problem.evaluate(x)

    def has_next(self):
        return self.counter < self.n_eval

    def notify(self, obj):
        self.data.append(obj)
