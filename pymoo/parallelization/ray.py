"""
Ray-based parallelization for pymoo.

This module provides parallelization using Ray distributed computing framework.
"""


class RayParallelization:
    """Use Ray as backend to parallelize problem evaluation.
    
    Ray is an open-source unified framework for scaling AI and Python applications.
    It provides distributed computing capabilities that can scale from a single
    machine to large clusters.
    
    Read more here: https://docs.ray.io
    
    Note
    ----
    You will need to install Ray to use this.
    Install with: pip install pymoo[parallelization] or pip install ray[default]
    
    Examples
    --------
    >>> import ray
    >>> ray.init()  # Initialize Ray
    >>> from pymoo.parallelization.ray import RayParallelization
    >>> runner = RayParallelization(job_resources={'num_cpus': 2})
    >>> # Use with problem.elementwise_runner = runner
    """
    
    def __init__(self, job_resources: dict = {'num_cpus': 1}) -> None:
        """
        Parameters
        ----------
        job_resources: dict, default: {'num_cpus': 1}
            A resource in Ray is a key-value pair where the key denotes a 
            resource name and the value is a float quantity. Ray has native support for CPU,
            GPU, and memory resource types; `'num_cpus'`, `'num_gpus'`, and `'memory'`.
            
            Examples:
            - {'num_cpus': 1}: Use 1 CPU per task
            - {'num_cpus': 2, 'num_gpus': 0.5}: Use 2 CPUs and half a GPU per task
            - {'memory': 1000 * 1024 * 1024}: Reserve 1GB memory per task
            
            Read more here: 
            https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        """
        # Check ray availability at runtime
        try:
            import ray as _ray
        except ImportError:
            raise ImportError(
                "Ray must be installed! "
                "You can install Ray with one of:\n"
                "  pip install pymoo[parallelization]\n"
                "  pip install pymoo[full]\n"
                '  pip install "ray[default]"'
            )
        self.ray = _ray
        super().__init__()
        self.job_resources = job_resources

    def __call__(self, f, X):
        """
        Execute function f on each element of X in parallel using Ray.
        
        Parameters
        ----------
        f : callable
            Function to apply to each element. Should be the evaluation function
            from an ElementwiseEvaluationFunction.
        X : list
            List of inputs to process
            
        Returns
        -------
        list
            Results from parallel execution using Ray
        """
        # Create a Ray remote function with specified resources
        runnable = self.ray.remote(f.__call__.__func__)
        runnable = runnable.options(**self.job_resources)
        
        # Submit all tasks and collect futures
        futures = [runnable.remote(f, x) for x in X]
        
        # Wait for all tasks to complete and return results
        return self.ray.get(futures)

    def __getstate__(self):
        """Handle pickling by preserving configuration parameters."""
        state = self.__dict__.copy()
        return state