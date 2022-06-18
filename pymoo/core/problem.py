from abc import abstractmethod

import autograd.numpy as np

from pymoo.util.cache import Cache
from pymoo.util.misc import at_least_2d_array


# ---------------------------------------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------------------------------------


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
                 check_inconsistencies=True,
                 replace_nan_values_by=np.inf,
                 exclude_from_serialization=None,
                 callback=None,
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

        if "elementwise_evaluation" in kwargs and kwargs.get("elementwise_evaluation"):
            raise Exception("The interface in pymoo 0.5.0 has changed. Please inherit from the ElementwiseProblem "
                            "class AND remove the 'elementwise_evaluation=True' argument to disable this exception.")

        # if variables are provided directly
        if vars is not None:
            n_var = len(vars)

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
        self.vars = vars

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

        # whether the problem should strictly be checked for inconsistency during evaluation
        self.check_inconsistencies = check_inconsistencies

        # this defines if NaN values should be replaced or not
        self.replace_nan_values_by = replace_nan_values_by

        # attribute which are excluded from being serialized )
        self.exclude_from_serialization = exclude_from_serialization if exclude_from_serialization is not None else []

        # the variable type
        self.vartype = vtype

    def evaluate(self,
                 X,
                 *args,
                 return_values_of=("F", "G", "H"),
                 return_as_dictionary=False,
                 **kwargs):

        # make sure the array is at least 2d. store if reshaping was necessary
        if isinstance(X, np.ndarray) and X.dtype != object:
            X, only_single_value = at_least_2d_array(X, extend_as="row", return_if_reshaped=True)
            assert X.shape[1] == self.n_var, f'Input dimension {X.shape[1]} are not equal to n_var {self.n_var}!'
        else:
            only_single_value = not (isinstance(X, list) or isinstance(X, np.ndarray))

        # do the actual evaluation for the given problem - calls in _evaluate method internally
        out = self.do(X, return_values_of, *args, **kwargs)

        # if enabled (recommended) the output shapes are checked for inconsistencies
        if self.check_inconsistencies:
            pass

        # if the NaN values should be replaced
        if self.replace_nan_values_by is not None:
            replace_nan_values(out, self.replace_nan_values_by)

        # make sure F and G are in fact floats (at least try to do that, no exception will be through if it fails)
        out_to_float(out, ["F", "G", "H"])

        # in case the input had only one dimension, then remove always the first dimension from each output
        if only_single_value:
            out_to_1d_ndarray(out)

        if self.callback is not None:
            self.callback(X, out)

        # now depending on what should be returned prepare the output
        if return_as_dictionary:
            return out
        else:
            if len(return_values_of) == 1:
                return out[return_values_of[0]]
            else:
                return tuple([out[e] for e in return_values_of])

    def do(self, X, return_values_of, *args, **kwargs):

        # prepare the dictionary to be filled after the evaluation
        out = dict_with_defaults(self, X, return_values_of)

        # do the actual evaluation in the problem definition
        self._evaluate(X, out, *args, **kwargs)

        # make sure all outputs are a 2d array
        out_to_2d_ndarray(out)

        return out

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


def calc_ps(problem, *args, **kwargs):
    return at_least_2d_array(problem._calc_pareto_set(*args, **kwargs))


def calc_pf(problem, *args, **kwargs):
    return at_least_2d_array(problem._calc_pareto_front(*args, **kwargs))


# ---------------------------------------------------------------------------------------------------------
# Elementwise Problem
# ---------------------------------------------------------------------------------------------------------


def elementwise_eval(problem, x, args, kwargs):
    out = dict()
    problem._evaluate(x, out, *args, **kwargs)
    return out


def looped_eval(func_elementwise_eval, problem, X, *args, **kwargs):
    return [func_elementwise_eval(problem, x, args, kwargs) for x in X]


def starmap_parallelized_eval(func_elementwise_eval, problem, X, *args, **kwargs):
    starmap = problem.runner
    params = [(problem, x, args, kwargs) for x in X]
    return list(starmap(func_elementwise_eval, params))


def dask_parallelized_eval(func_elementwise_eval, problem, X, *args, **kwargs):
    client = problem.runner
    jobs = [client.submit(func_elementwise_eval, problem, x, args, kwargs) for x in X]
    return [job.result() for job in jobs]


class ElementwiseProblem(Problem):

    def __init__(self,
                 func_elementwise_eval=elementwise_eval,
                 func_eval=looped_eval,
                 exclude_from_serialization=None,
                 runner=None,
                 **kwargs):

        super().__init__(exclude_from_serialization=exclude_from_serialization, **kwargs)

        # the most granular function which evaluates one single individual - this is the function to parallelize
        self.func_elementwise_eval = func_elementwise_eval

        # the function that calls func_elementwise_eval for ALL solutions to be evaluated
        self.func_eval = func_eval

        # the two ways of parallelization which are supported
        self.runner = runner

        # do not serialize the starmap - this will throw an exception
        self.exclude_from_serialization = self.exclude_from_serialization + ["runner"]

    def do(self, X, return_values_of, *args, **kwargs):
        out = dict_with_defaults(self, X, return_values_of)

        # do an elementwise evaluation and return the results
        ret = self.func_eval(self.func_elementwise_eval, self, X, *args, **kwargs)

        # all the elements to be set to the output
        keys = list(ret[0].keys())

        for key in keys:
            if key not in out:
                out[key] = [None] * len(X)

        # for each evaluation call
        for i, elem in enumerate(ret):
            for k, v in elem.items():
                try:
                    out[k][i] = v
                except Exception as e:
                    raise Exception(f"Problem Error: {k} can not be set", e)

        # make sure the array is 2d before doing the shape check
        out_to_2d_ndarray(out)

        return out

    @abstractmethod
    def _evaluate(self, x, out, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------------------------------------

def default_return_values(has_constr=False):
    vals = ["F"]
    if has_constr:
        vals.append("CV")
    return vals


def dict_with_none(keys):
    out = {}
    for val in keys:
        out[val] = None
    return out


def defaults_of_out(problem, k):
    DEFAULTS = dict(
        F=lambda: np.full((k, problem.n_obj), np.nan, dtype=float),
        G=lambda: np.full((k, problem.n_ieq_constr), np.nan, dtype=float),
        H=lambda: np.full((k, problem.n_eq_constr), np.nan, dtype=float),
    )
    return DEFAULTS


def dict_with_defaults(problem, X, keys):
    DEFAULTS = defaults_of_out(problem, len(X))

    out = {}
    for key in keys:
        if key in DEFAULTS:
            out[key] = DEFAULTS[key]()
        else:
            out[key] = None
    return out


def out_to_ndarray(out):
    for key, val in out.items():
        if val is not None:
            if not isinstance(val, np.ndarray):
                out[key] = np.array([val])


def out_to_2d_ndarray(out):
    for key, val in out.items():
        if val is not None:
            if isinstance(val, np.ndarray):
                if val.ndim == 1:
                    out[key] = val[:, None]


def out_to_1d_ndarray(out):
    for key in out.keys():
        if out[key] is not None:
            out[key] = out[key][0, :]


def out_to_float(out, keys):
    for key in keys:
        if key in out:
            try:
                out[key] = out[key].astype(float)
            except:
                pass


def replace_nan_values(out, by=np.inf):
    for key in out:
        try:
            v = out[key]
            v[np.isnan(v)] = by
            out[key] = v
        except:
            pass

