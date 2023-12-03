import numpy as np

from pymoo.core.observer import Observer


def f_gap(a):
    best = a.opt.solution()
    if best.feas:
        return a.opt.solution().f - a.problem.fopt()


def f_min(a):
    best = a.opt.solution()
    if best.feas:
        return best.f


def default_columns():
    return {
        'iter': Column(name='iter', f_get=lambda a: a.iter),
        'fevals': Column(name='fevals', f_get=lambda a: a.evaluator.fevals),
        'f_min': Column(name='f_min', f_get=f_min),
        'cv_min': Column(name='cv_min',
                         f_get=lambda a: a.opt.solution().cv,
                         f_avail=lambda a: a.problem.ptype.has_constraints()),
        'f_gap': Column(name='f_gap',
                       f_get=f_gap,
                       f_avail=lambda a: a.problem.fopt() is not None)
    }


class Output(Observer):

    def __init__(self, columns: dict = None):
        super().__init__()
        self.columns = columns

    def setup(self, _):
        pass

    def initialize(self, algorithm):
        print(self.header(border=True))
        self.update(algorithm)

    def update(self, algorithm):
        try:
            [col.update(algorithm) for col in self.columns]
            text = self.text()
        except Exception as e:
            text = f"Display Error: {e}"

        print(text)

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


class SingleObjectiveOutput(Output):

    def setup(self, algorithm):
        labels = ["iter", "fevals", "cv_min", "f_min", "f_gap"]

        options = default_columns()
        columns = [options.get(key) for key in labels]

        self.columns = [column for column in columns if column.is_available(algorithm)]
        return self


class Column:

    def __init__(self,
                 name,
                 width=13,
                 f_get=None,
                 f_avail=lambda _: True,
                 truncate=True) -> None:
        super().__init__()
        self.name = name
        self.f_get = f_get
        self.f_avail = f_avail
        self.width = width
        self.truncate = truncate
        self.value = None

    def is_available(self, algorithm):
        if self.f_avail is not None:
            return self.f_avail(algorithm)
        else:
            return True

    def update(self, algorithm):

        value = None

        if self.is_available(algorithm):
            if self.f_get is not None:
                try:
                    value = self.f_get(algorithm)
                except:
                    value = None

        if value is None:
            value = '-'
        self.value = value

    def header(self):
        return str(self.name).center(self.width)

    def text(self):
        value = self.value
        if value is None:
            value = "-"

        return format_text(value, self.width, self.truncate)

    def set(self, value):
        self.value = value


def number_to_text(number, width):
    if number >= 10 or number * 1e5 < 1:
        return f"%.{width - 7}E" % number
    else:
        return f"%.{width - 3}f" % number


def format_text(value, width, truncate):
    if value is not None:

        if isinstance(value, (np.floating, float)):
            text = number_to_text(value, width)
        else:
            text = str(value)

        if truncate and len(text) > width:
            text = text[:width]
    else:
        text = "-"
    text = text.rjust(width)
    return text
