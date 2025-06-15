"""
Joblib-based parallelization for pymoo.

This module provides comprehensive parallelization using joblib with full configuration support.
"""

from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Literal

# Imports will be done at runtime to avoid circular imports


class JoblibParallelization:
    """Parallelization using joblib.
    
    This class provides a comprehensive interface to joblib's parallel execution
    capabilities with full control over all joblib.Parallel parameters.
    
    Note
    ----
    You will need to install `joblib` (and optionally `psutil`) to use this.
    Install with: pip install pymoo[parallelization]
    """

    def __init__(
        self,
        n_jobs: int = -1,
        backend: Literal["loky", "threading", "multiprocessing"] = "loky",
        return_as: Literal["list", "generator"] = "list",
        verbose: int = 0,
        timeout: float | None = None,
        pre_dispatch: str | int = "2 * n_jobs",
        batch_size: int | Literal["auto"] = "auto",
        temp_folder: str | Path | None = None,
        max_nbytes: int | str | None = "1M",
        mmap_mode: Literal["r+", "r", "w+", "c"] | None = "r",
        prefer: Literal["processes", "threads"] | None = None,
        require: Literal["sharedmem"] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        n_jobs: int, default: -1
            The maximum number of concurrently running jobs, such as the number
            of Python worker processes when backend="multiprocessing"
            or the size of the thread-pool when backend="threading".
            If -1 all CPUs are used.
            If 1 is given, no parallel computing code is used at all, and the
            behavior amounts to a simple python ``for`` loop. This mode is not
            compatible with ``timeout``.
            For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
            n_jobs = -2, all CPUs but one are used.
            None is a marker for 'unset' that will be interpreted as n_jobs=1
            unless the call is performed under a :func:`~parallel_config`
            context manager that sets another value for ``n_jobs``.
        backend: str, default: 'loky'
            Specify the parallelization backend implementation.
            Supported backends are:

            - "loky" used by default, can induce some
                communication and memory overhead when exchanging input and
                output data with the worker Python processes. On some rare
                systems (such as Pyiodide), the loky backend may not be
                available.
            - "multiprocessing" previous process-based backend based on
                ``multiprocessing.Pool``. Less robust than ``loky``.
            - "threading" is a very low-overhead backend but it suffers
                from the Python Global Interpreter Lock if the called function
                relies a lot on Python objects. "threading" is mostly useful
                when the execution bottleneck is a compiled extension that
                explicitly releases the GIL (for instance a Cython loop wrapped
                in a "with nogil" block or an expensive call to a library such
                as NumPy).

            It is not recommended to hard-code the backend name in a call to
            :class:`~Parallel` in a library. Instead it is recommended to set
            soft hints (prefer) or hard constraints (require) so as to make it
            possible for library users to change the backend from the outside
            using the :func:`~parallel_config` context manager.
        return_as: {'list', 'generator'}, default: 'list'
            If 'list', calls to this instance will return a list, only when
            all results have been processed and retrieved.
            If 'generator', it will return a generator that yields the results
            as soon as they are available, in the order the tasks have been
            submitted with.
            Future releases are planned to also support 'generator_unordered',
            in which case the generator immediately yields available results
            independently of the submission order.
        prefer: {'processes', 'threads'} or None, default: None
            Soft hint to choose the default backend if no specific backend
            was selected with the :func:`~parallel_config` context manager.
            The default process-based backend is 'loky' and the default
            thread-based backend is 'threading'. Ignored if the ``backend``
            parameter is specified.
        require: 'sharedmem' or None, default: None
            Hard constraint to select the backend. If set to 'sharedmem',
            the selected backend will be single-host and thread-based even
            if the user asked for a non-thread based backend with
            :func:`~joblib.parallel_config`.
        verbose: int, optional, default: 0
            The verbosity level: if non zero, progress messages are
            printed. Above 50, the output is sent to stdout.
            The frequency of the messages increases with the verbosity level.
            If it more than 10, all iterations are reported.
        timeout: float, optional, default: None
            Timeout limit for each task to complete.  If any task takes longer
            a TimeOutError will be raised. Only applied when n_jobs != 1
        pre_dispatch: {'all', integer, or expression, as in '3*n_jobs'}, default: '2*n_jobs'
            The number of batches (of tasks) to be pre-dispatched.
            Default is '2*n_jobs'. When batch_size="auto" this is reasonable
            default and the workers should never starve. Note that only basic
            arithmetic are allowed here and no modules can be used in this
            expression.
        batch_size: int or 'auto', default: 'auto'
            The number of atomic tasks to dispatch at once to each
            worker. When individual evaluations are very fast, dispatching
            calls to workers can be slower than sequential computation because
            of the overhead. Batching fast computations together can mitigate
            this.
            The ``'auto'`` strategy keeps track of the time it takes for a
            batch to complete, and dynamically adjusts the batch size to keep
            the time on the order of half a second, using a heuristic. The
            initial batch size is 1.
            ``batch_size="auto"`` with ``backend="threading"`` will dispatch
            batches of a single task at a time as the threading backend has
            very little overhead and using larger batch size has not proved to
            bring any gain in that case.
        temp_folder: str, optional, default: None
            Folder to be used by the pool for memmapping large arrays
            for sharing memory with worker processes. If None, this will try in
            order:

            - a folder pointed by the JOBLIB_TEMP_FOLDER environment
                variable,
            - /dev/shm if the folder exists and is writable: this is a
                RAM disk filesystem available by default on modern Linux
                distributions,
            - the default system temporary folder that can be
                overridden with TMP, TMPDIR or TEMP environment
                variables, typically /tmp under Unix operating systems.

            Only active when backend="loky" or "multiprocessing".
        max_nbytes int, str, or None, optional, default: '1M'
            Threshold on the size of arrays passed to the workers that
            triggers automated memory mapping in temp_folder. Can be an int
            in Bytes, or a human-readable string, e.g., '1M' for 1 megabyte.
            Use None to disable memmapping of large arrays.
            Only active when backend="loky" or "multiprocessing".
        mmap_mode: {'r+', 'r', 'w+', 'c'} or None, default: 'r'
            Memmapping mode for numpy arrays passed to workers. None will
            disable memmapping, other modes defined in the numpy.memmap doc:
            https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
            Also, see 'max_nbytes' parameter documentation for more details.
        
        Examples
        --------
        >>> import os
        >>> from pymoo.core.problem import ElementwiseProblem
        >>> from pymoo.optimize import minimize
        >>> from pymoo.algorithms.soo.nonconvex.ga import GA

        >>> class MyProblem(ElementwiseProblem):
        >>>     def __init__(self, **kwargs):
        >>>         super().__init__(n_var=10, n_obj=1, n_ieq_constr=0, xl=-5, xu=5, **kwargs)

        >>>     def _evaluate(self, X, out, *args, **kwargs):
        >>>         out["F"] = (X ** 2).sum()

        >>> from pymoo.parallelization.joblib import JoblibParallelization
        >>> runner = JoblibParallelization()
        >>> problem = MyProblem(runner=runner)
        >>> res = minimize(problem, GA(), termination=("n_gen", 200), seed=1)
        >>> print(f'Joblib runtime: {res.exec_time:.2f} sec with {os.cpu_count()} cores')
        Joblib runtime: 0.54 sec with 12 cores
        """
        # Check joblib availability at runtime
        try:
            import joblib as _joblib
        except ImportError:
            msg = (
                "joblib must be installed! "
                "You can install joblib with one of:\n"
                "  pip install pymoo[parallelization]\n"
                "  pip install pymoo[full]\n"
                "  pip install joblib"
            )
            raise ImportError(msg)
        self.joblib = _joblib
        self.n_jobs = n_jobs
        self.backend = backend
        self.return_as = return_as
        self.verbose = verbose
        self.timeout = timeout
        self.pre_dispatch = pre_dispatch
        self.batch_size = batch_size
        self.temp_folder = temp_folder
        self.max_nbytes = max_nbytes
        self.mmap_mode = mmap_mode
        self.prefer = prefer
        self.require = require
        super().__init__()

    def __call__(
        self,
        f: Callable[..., Any],
        X: Iterable[Any],
    ) -> list[Any] | Generator[Any, Any, None]:
        """
        Execute function f on each element of X in parallel using joblib.
        
        Parameters
        ----------
        f : callable
            Function to apply to each element
        X : iterable
            Iterable of inputs to process
            
        Returns
        -------
        list or generator
            Results from parallel execution, type depends on return_as parameter
        """
        with self.joblib.Parallel(
            n_jobs=self.n_jobs,
            backend=self.backend,
            return_as=self.return_as,
            verbose=self.verbose,
            timeout=self.timeout,
            pre_dispatch=self.pre_dispatch,
            batch_size=self.batch_size,
            temp_folder=self.temp_folder,
            max_nbytes=self.max_nbytes,
            mmap_mode=self.mmap_mode,
            prefer=self.prefer,
            require=self.require,
        ) as parallel:
            return parallel(self.joblib.delayed(f)(x) for x in X)

    def __getstate__(self):
        """Handle pickling by preserving configuration parameters."""
        state = self.__dict__.copy()
        # Note: We keep all parameters since they're needed for reconstruction
        return state