from pymoo.core.observer import Observer


class Display(Observer):

    def __init__(self):
        super().__init__()

    def notify(self, algorithm, event):
        if algorithm.verbose and algorithm.output is not None:
            algorithm.output.notify(algorithm, event)


