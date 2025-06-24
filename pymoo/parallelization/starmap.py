"""
Starmap-based parallelization for pymoo.
"""


class StarmapParallelization:
    """Parallelization using a starmap function.
    
    Parameters
    ----------
    starmap : callable
        A starmap function like multiprocessing.Pool.starmap
    """

    def __init__(self, starmap):
        self.starmap = starmap

    def __call__(self, f, X):
        return list(self.starmap(f, [[x] for x in X]))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("starmap", None)
        return state