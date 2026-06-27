"""Ray-based parallelization for pymoo."""


class RayParallelization:
    """Parallelization using Ray.

    Args:
        job_resources: Ray resource requirements per job (default: {'num_cpus': 1}).
    """

    def __init__(self, job_resources: dict | None = None) -> None:
        if job_resources is None:
            job_resources = {"num_cpus": 1}
        try:
            import ray
        except ImportError:
            raise ImportError("Ray must be installed: pip install ray")

        self.ray = ray
        self.job_resources = job_resources

    def __call__(self, f, X):
        runnable = self.ray.remote(f.__call__.__func__)
        runnable = runnable.options(**self.job_resources)
        futures = [runnable.remote(f, x) for x in X]
        return self.ray.get(futures)

    def __getstate__(self):
        return self.__dict__.copy()
