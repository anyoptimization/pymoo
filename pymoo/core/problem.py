from abc import abstractmethod

import numpy as np

import pymoo.gradient.toolbox as anp
from pymoo.util.cache import Cache
from pymoo.util.misc import at_least_2d_array

try:
    import ray
except ImportError:
    ray = None


class ElementwiseEvaluationFunction:

    def __init__(self, problem, args, kwargs) -> None:
        super().__init__()
        self.problem = problem
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        out = dict()
        self.problem._evaluate(x, out, *self.args, **self.kwargs)
        return out


class LoopedElementwiseEvaluation:

    def __call__(self, f, X):
        return [f(x) for x in X]


class StarmapParallelization:

    def __init__(self, starmap) -> None:
        super().__init__()
        self.starmap = starmap

    def __call__(self, f, X):
        return list(self.starmap(f, [[x] for x in X]))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("starmap", None)
        return state


class DaskParallelization:

    def __init__(self, client) -> None:
        super().__init__()
        self.client = client

    def __call__(self, f, X):
        jobs = [self.client.submit(f, x) for x in X]
        return [job.result() for job in jobs]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("client", None)
        return state


class JoblibParallelization:

    def __init__(self, aJoblibParallel,  aJoblibDelayed, *args, **kwargs) -> None:
        super().__init__()
        self.parallel = aJoblibParallel 
        self.delayed =  aJoblibDelayed 

    def __call__(self, f, X):
        return self.parallel(self.delayed(f)(x) for x in X)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("parallel", None)
        state.pop("delayed", None)
        return state


class RayParallelization:
    """Use Ray as backend to parallelize problem evaluation.
    
    Ray is an open-source unified framework for scaling AI and Python applicaitons.
    Read more here: https://docs.ray.io.
    
    You will need to install Ray to use this.
    """
    def __init__(self, job_resources: dict = {'num_cpus': 1}) -> None:
        """
        Parameters
        ----------
        job_resources: A resource in Ray is a key-value pair where the key denotes a 
            resource name and the value is a float quantity. Ray has native support for CPU,
            GPU, and memory resource types; `'num_cpus'`, `'num_gpus'`, and `'memory'`.
            Read more here: 
            https://docs.ray.io/en/latest/ray-core/scheduling/resources.html.
        """
        assert ray is not None, (
            "Ray must be installed! "
            "You can install Ray with the command: "
            '`pip install -U "ray[default]"`'
        )
        super().__init__()
        self.job_resources = job_resources

    def __call__(self, f, X):
        runnable = ray.remote(f.__call__.__func__)
        runnable = runnable.options(**self.job_resources)
        futures = [runnable.remote(f, x) for x in X]
        return ray.get(futures)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state


class Problem:
    def __init__(self,
                 n_var=-1,
                 n_obj=1,
                 n_ieq_constr=0,
                 n_eq_constr=0,
                 xl=None,
                 xu=None,
                 vtype=None,
                 vars=None,
                 elementwise=False,
                 elementwise_func=ElementwiseEvaluationFunction,
                 elementwise_runner=LoopedElementwiseEvaluation(),
                 requires_kwargs=False,
                 replace_nan_values_by=None,
                 exclude_from_serialization=None,
                 callback=None,
                 strict=True,
                 **kwargs):

        """

        Parameters
        ----------
        n_var : int
            Number of Variables

        n_obj : int
            Number of Objectives

        n_ieq_constr : int
            Number of Inequality Constraints

        n_eq_constr : int
            Number of Equality Constraints

        xl : np.array, float, int
            Lower bounds for the variables. if integer all lower bounds are equal.

        xu : np.array, float, int
            Upper bounds for the variable. if integer all upper bounds are equal.

        vtype : type
            The variable type. So far, just used as a type hint.

        """

        # number of variable
        self.n_var = n_var

        # number of objectives
        self.n_obj = n_obj

        # number of inequality constraints
        self.n_ieq_constr = n_ieq_constr if "n_constr" not in kwargs else max(n_ieq_constr, kwargs["n_constr"])

        # number of equality constraints
        self.n_eq_constr = n_eq_constr

        # type of the variable to be evaluated
        self.data = dict(**kwargs)

        # the lower bounds, make sure it is a numpy array with the length of n_var
        self.xl, self.xu = xl, xu

        # a callback function to be called after every evaluation
        self.callback = callback

        # if the variables are provided in their explicit form
        if vars is not None:
            self.vars = vars
            self.n_var = len(vars)

            if self.xl is None:
                self.xl = {name: var.lb if hasattr(var, "lb") else None for name, var in vars.items()}
            if self.xu is None:
                self.xu = {name: var.ub if hasattr(var, "ub") else None for name, var in vars.items()}

        # the variable type (only as a type hint at this point)
        self.vtype = vtype

        # the functions used if elementwise is enabled
        self.elementwise = elementwise
        self.elementwise_func = elementwise_func
        self.elementwise_runner = elementwise_runner

        # whether evaluation requires kwargs (passing them can cause overhead in parallelization)
        self.requires_kwargs = requires_kwargs

        # whether the shapes are checked strictly
        self.strict = strict

        # if it is a problem with an actual number of variables - make sure xl and xu are numpy arrays
        if n_var > 0:

            if self.xl is not None:
                if not isinstance(self.xl, np.ndarray):
                    self.xl = np.ones(n_var) * xl
                self.xl = self.xl.astype(float)

            if self.xu is not None:
                if not isinstance(self.xu, np.ndarray):
                    self.xu = np.ones(n_var) * xu
                self.xu = self.xu.astype(float)

        # this defines if NaN values should be replaced or not
        self.replace_nan_values_by = replace_nan_values_by

        # attribute which are excluded from being serialized
        self.exclude_from_serialization = exclude_from_serialization

    def evaluate(self,
                 X,
                 *args,
                 return_values_of=None,
                 return_as_dictionary=False,
                 **kwargs):

        # if the problem does not require any kwargs they are re-initialized
        if not self.requires_kwargs:
            kwargs = dict()

        if return_values_of is None:
            return_values_of = ["F"]
            if self.n_ieq_constr > 0:
                return_values_of.append("G")
            if self.n_eq_constr > 0:
                return_values_of.append("H")

        # make sure the array is at least 2d. store if reshaping was necessary
        if isinstance(X, np.ndarray) and X.dtype != object:
            X, only_single_value = at_least_2d_array(X, extend_as="row", return_if_reshaped=True)
            assert X.shape[1] == self.n_var, f'Input dimension {X.shape[1]} are not equal to n_var {self.n_var}!'
        else:
            only_single_value = not (isinstance(X, list) or isinstance(X, np.ndarray))

        # this is where the actual evaluation takes place
        _out = self.do(X, return_values_of, *args, **kwargs)

        out = {}
        for k, v in _out.items():

            # copy it to a numpy array (it might be one of jax at this point)
            v = np.array(v)

            # in case the input had only one dimension, then remove always the first dimension from each output
            if only_single_value:
                v = v[0]

            # if the NaN values should be replaced
            if self.replace_nan_values_by is not None:
                v[np.isnan(v)] = self.replace_nan_values_by

            try:
                out[k] = v.astype(np.float64)
            except:
                out[k] = v

        if self.callback is not None:
            self.callback(X, out)

        # now depending on what should be returned prepare the output
        if return_as_dictionary:
            return out

        if len(return_values_of) == 1:
            return out[return_values_of[0]]
        else:
            return tuple([out[e] for e in return_values_of])

    def do(self, X, return_values_of, *args, **kwargs):

        # create an empty dictionary
        out = {name: None for name in return_values_of}

        # do the function evaluation
        if self.elementwise:
            self._evaluate_elementwise(X, out, *args, **kwargs)
        else:
            self._evaluate_vectorized(X, out, *args, **kwargs)

        # finally format the output dictionary
        out = self._format_dict(out, len(X), return_values_of)

        return out

    def _evaluate_vectorized(self, X, out, *args, **kwargs):
        self._evaluate(X, out, *args, **kwargs)

    def _evaluate_elementwise(self, X, out, *args, **kwargs):

        # create the function that evaluates a single individual
        f = self.elementwise_func(self, args, kwargs)

        # execute the runner
        elems = self.elementwise_runner(f, X)

        # for each evaluation call
        for elem in elems:

            # for each key stored for this evaluation
            for k, v in elem.items():

                # if the element does not exist in out yet -> create it
                if out.get(k, None) is None:
                    out[k] = []

                out[k].append(v)

        # convert to arrays (the none check is important because otherwise an empty array is initialized)
        for k in out:
            if out[k] is not None:
                out[k] = anp.array(out[k])

    def _format_dict(self, out, N, return_values_of):

        # get the default output shape for the default values
        shape = default_shape(self, N)

        # finally the array to be returned
        ret = {}

        # for all values that have been set in the user implemented function
        for name, v in out.items():

            # only if they have truly been set
            if v is not None:

                # if there is a shape to be expected
                if name in shape:

                    if isinstance(v, list):
                        v = anp.column_stack(v)

                    try:
                        v = v.reshape(shape[name])
                    except Exception as e:
                        raise Exception(
                            f"Problem Error: {name} can not be set, expected shape {shape[name]} but provided {v.shape}",
                            e)

                ret[name] = v

        # if some values that are necessary have not been set
        for name in return_values_of:
            if name not in ret:
                s = shape.get(name, N)
                ret[name] = np.full(s, np.inf)

        return ret

    @Cache
    def nadir_point(self, *args, **kwargs):
        pf = self.pareto_front(*args, **kwargs)
        if pf is not None:
            return np.max(pf, axis=0)

    @Cache
    def ideal_point(self, *args, **kwargs):
        pf = self.pareto_front(*args, **kwargs)
        if pf is not None:
            return np.min(pf, axis=0)

    @Cache
    def pareto_front(self, *args, **kwargs):
        pf = self._calc_pareto_front(*args, **kwargs)
        pf = at_least_2d_array(pf, extend_as='r')
        if pf is not None and pf.shape[1] == 2:
            pf = pf[np.argsort(pf[:, 0])]
        return pf

    @Cache
    def pareto_set(self, *args, **kwargs):
        ps = self._calc_pareto_set(*args, **kwargs)
        ps = at_least_2d_array(ps, extend_as='r')
        return ps

    @property
    def n_constr(self):
        return self.n_ieq_constr + self.n_eq_constr

    @abstractmethod
    def _evaluate(self, x, out, *args, **kwargs):
        pass

    def has_bounds(self):
        return self.xl is not None and self.xu is not None

    def has_constraints(self):
        return self.n_constr > 0

    def bounds(self):
        return self.xl, self.xu

    def name(self):
        return self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        pass

    def _calc_pareto_set(self, *args, **kwargs):
        pass

    def __str__(self):
        s = "# name: %s\n" % self.name()
        s += "# n_var: %s\n" % self.n_var
        s += "# n_obj: %s\n" % self.n_obj
        s += "# n_ieq_constr: %s\n" % self.n_ieq_constr
        s += "# n_eq_constr: %s\n" % self.n_eq_constr
        return s

    def __getstate__(self):
        if self.exclude_from_serialization is not None:
            state = self.__dict__.copy()

            # exclude objects which should not be stored
            for key in self.exclude_from_serialization:
                state[key] = None

            return state
        else:
            return self.__dict__


class ElementwiseProblem(Problem):

    def __init__(self, elementwise=True, **kwargs):
        super().__init__(elementwise=elementwise, **kwargs)


def default_shape(problem, n):
    n_var = problem.n_var
    DEFAULTS = dict(
        F=(n, problem.n_obj),
        G=(n, problem.n_ieq_constr),
        H=(n, problem.n_eq_constr),
        dF=(n, problem.n_obj, n_var),
        dG=(n, problem.n_ieq_constr, n_var),
        dH=(n, problem.n_eq_constr, n_var),
    )
    return DEFAULTS
