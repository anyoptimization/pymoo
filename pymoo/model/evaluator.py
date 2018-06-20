class Evaluator:

    def __init__(self, n_eval=10000):
        self.n_eval = n_eval
        self.counter = 0

    def eval(self, problem, x, **kwargs):
        if len(x.shape) == 1:
            self.counter += 1
        else:
            self.counter += x.shape[0]
        return problem.evaluate(x, **kwargs)

    def count_left(self):
        return self.n_eval - self.counter

    def has_next(self):
        return self.counter < self.n_eval
