import math

from alive_progress import alive_bar


class ProgressBar:

    def __init__(self, *args, start=True, non_decreasing=True, **kwargs):
        self.args = args
        self.kwargs = kwargs

        for key, default in [("manual", True), ("force_tty", True)]:
            if key not in kwargs:
                kwargs[key] = default

        self.func = None
        self.obj = None
        self.non_decreasing = non_decreasing
        self._max = 0.0

        if start:
            self.start()

    def set(self, value, *args, **kwargs):
        if self.non_decreasing:
            self._max = max(self._max, value)
            value = self._max

        prec = 100
        value = math.floor(value * prec) / prec

        self.obj(value, *args, **kwargs)

    def start(self):

        if not self.obj:
            # save the generator to this object
            self.func = alive_bar(*self.args, **self.kwargs).gen

            # create the bar
            self.obj = next(self.func)

    def close(self):
        if self.obj:
            try:
                next(self.func)
            except:
                pass

    def __enter__(self):
        self.start()

    def __exit__(self, type, value, traceback):
        self.close()
