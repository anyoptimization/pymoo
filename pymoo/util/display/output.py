from pymoo.core.callback import Callback
from pymoo.util.display.column import Column


def pareto_front_if_possible(problem):
    try:
        return problem.pareto_front()
    except:
        return None


class NumberOfGenerations(Column):

    def __init__(self, **kwargs) -> None:
        super().__init__("n_gen", **kwargs)

    def update(self, algorithm):
        self.value = algorithm.n_gen


class NumberOfEvaluations(Column):

    def __init__(self, **kwargs) -> None:
        super().__init__("n_eval", **kwargs)

    def update(self, algorithm):
        self.value = algorithm.evaluator.n_eval


class Output(Callback):

    def __init__(self):
        super().__init__()
        self.n_gen = NumberOfGenerations(width=6)
        self.n_eval = NumberOfEvaluations(width=8)
        self.columns = [self.n_gen, self.n_eval]

    def update(self, algorithm):
        [col.update(algorithm) for col in self.columns]

    def header(self, border=False):
        regex = " | ".join(["{}"] * len(self.columns))
        header = regex.format(*[col.name.center(col.width) for col in self.columns])

        if border:
            line = "=" * len(header)
            header = line + '\n' + header + '\n' + line

        return header

    def text(self):
        regex = " | ".join(["{}"] * len(self.columns))
        return regex.format(*[col.text() for col in self.columns])
