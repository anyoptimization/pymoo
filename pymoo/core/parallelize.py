from abc import abstractmethod
from multiprocessing import Pool as ProcessPool
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from joblib import Parallel, delayed
from tqdm import tqdm


# ---------------------------------------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------------------------------------

class Parallelize(object):

    def __call__(self,
                 func,
                 iterable,
                 total=None
                 ) -> list:
        return self.run(func, iterable)

    @abstractmethod
    def run(self, func, iterable, total=None):
        """
        A parallelization method to run multiple methods in parallel.

        Parameters
        ----------
        func
            The function which takes exactly one argument

        iterable
            A generator or list object for which to run the function.

        total
            The number of elements returned by the generator (used for the progress bar if implemented)

        Returns
        -------
        results
            A list of objects returned in each function call.

        """

        pass


# ---------------------------------------------------------------------------------------------------------
# Serial (not in parallel, simply using a for loop)
# ---------------------------------------------------------------------------------------------------------

class Serial(Parallelize):

    def __init__(self, progress=False) -> None:
        super().__init__()
        self.progress = progress

    def run(self, func, iterable, total=None):

        if self.progress:

            if total is None:
                iterable = list(iterable)
                total = len(iterable)

            iterable = list(iterable)
            return [func(args) for args in tqdm(iterable, total=total)]
        else:
            return [func(args) for args in iterable]


# ---------------------------------------------------------------------------------------------------------
# Pool (using Python's multithread/multiprocess)
# ---------------------------------------------------------------------------------------------------------

class Pool(Parallelize):

    def __init__(self, pool=None, mode="thread", max_workers=None, progress=False) -> None:
        super().__init__()

        if pool is None:

            if max_workers is None:
                max_workers = cpu_count() - 1

            if mode.startswith("process"):
                pool = ProcessPool(max_workers)
            elif mode.startswith("thread"):
                pool = ThreadPool(max_workers)
            else:
                raise Exception("Unknown mode. Either `process` or `thread`.")

        self.pool = pool
        self.progress = progress

    def run(self, func, iterable, total=None):
        if self.progress:

            if total is None:
                iterable = list(iterable)
                total = len(iterable)

            futures = self.pool.imap(func, iterable)
            return [e for e in tqdm(futures, total=total)]
        else:
            return self.pool.map(func, iterable)


# ---------------------------------------------------------------------------------------------------------
# Joblib (third-party library)
# ---------------------------------------------------------------------------------------------------------

class Joblib(Parallelize):

    def __init__(self, n_jobs=-2, prefer="threads", progress=False) -> None:
        super().__init__()
        self.pool = Parallel(n_jobs=n_jobs, prefer=prefer, return_as="generator")
        self.progress = progress

    def run(self, func, iterable, total=None):
        if self.progress:

            if total is None:
                iterable = list(iterable)
                total = len(iterable)

            futures = self.pool(delayed(func)(e) for e in iterable)

            return [e for e in tqdm(futures, total=total)]
        else:
            return list(self.pool(delayed(func)(e) for e in iterable))