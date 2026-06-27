"""Joblib-based parallelization for pymoo."""


class JoblibParallelization:
    """Parallelization using joblib.

    Args:
        n_jobs: Number of parallel jobs. -1 uses all available cores (default: -1).
        **kwargs: Additional arguments passed to joblib.Parallel.
    """

    def __init__(self, n_jobs: int = -1, **kwargs: object) -> None:
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib must be installed: pip install joblib")

        self.joblib = joblib
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def __call__(self, f, X):
        with self.joblib.Parallel(n_jobs=self.n_jobs, **self.kwargs) as parallel:
            return parallel(self.joblib.delayed(f)(x) for x in X)

    def __getstate__(self):
        return self.__dict__.copy()
