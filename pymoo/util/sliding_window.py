import numpy as np


class SlidingWindow(list):

    def __init__(self, size=None) -> None:
        super().__init__()
        self.size = size

    def append(self, entry):
        super().append(entry)

        if self.size is not None:
            while len(self) > self.size:
                self.pop(0)

    def is_full(self):
        return self.size == len(self)

    def to_numpy(self):
        return np.array(self)
