import numpy as np


class Column:

    def __init__(self, name, width=13, func=None, truncate=True) -> None:
        super().__init__()
        self.name = name
        self.func = func
        self.width = width
        self.truncate = truncate
        self.value = None

    def update(self, algorithm):
        if self.func:
            self.value = self.func(algorithm)

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
