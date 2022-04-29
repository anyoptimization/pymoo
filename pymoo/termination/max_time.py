import time

from pymoo.core.termination import Termination
from pymoo.util.misc import time_to_int


class TimeBasedTermination(Termination):

    def __init__(self, max_time) -> None:
        super().__init__()
        self.start = None

        if isinstance(max_time, str):
            self.max_time = time_to_int(max_time)
        elif isinstance(max_time, int) or isinstance(max_time, float):
            self.max_time = max_time
        else:
            raise Exception("Either provide the time as a string or an integer.")

    def setup(self, _):
        self.start = time.time()

    def _update(self, algorithm):
        elapsed = time.time() - self.start
        return elapsed / self.max_time
