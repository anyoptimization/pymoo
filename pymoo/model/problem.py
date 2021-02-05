from abc import abstractmethod

import autograd.numpy as np

from pymoo.util.misc import at_least_2d_array


# ---------------------------------------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------------------------------------

class Problem:
    def __init__(self,
                 n_var=-1,
                 n_obj=1,
                 n_constr=0,
                 xl=None,
                 xu=None,
                 check_inconsistencies=True,
                 replace_nan_values_by=np.inf,
                 exclude_from_serialization=None,
                 **kwargs):

        """

        Parameters
        ----------
        n_var : int
            Number of Variables

        n_obj : int
            Number of Objectives

        n_constr : int
            Number of Constraints

        xl : np.array, float, int
            Lower bounds for the variables. if integer all lower bounds are equal.

        xu : np.array, float, int
            Upper bounds for the variable. if integer all upper bounds are equal.

        type_var : numpy.dtype


        """

        if "elementwise_evaluation" in kwargs and kwargs.get("elementwise_evaluation"):
            raise Exception("The interface in pymoo 0.5.0 has changed. Please inherit from the ElementwiseProblem "
                            "class AND remove the 'elementwise_evaluation=True' argument to disable this exception.")

        # number of variable
        self.n_var = n_var

        # number of objectives
        self.n_obj = n_obj

        # number of constraints
        self.n_constr = n_constr

        # type of the variable to be evaluated
        self.data = dict(**kwargs)

        # the lower bounds, make sure it is a numpy array with the length of n_var
        self.xl, self.xu = xl, xu

        # if it is a problem with an actual number of variables - make sure xl and xu are numpy arrays
        if n_var > 0:
            if self.xl is not None and not isinstance(self.xl, np.ndarray):
                self.xl = np.ones(n_var) * xl
            self.xu = xu
            if self.xu is not None and not isinstance(self.xu, np.ndarray):
                self.xu = np.ones(n_var) * xu

        # whether the problem should strictly be checked for inconsistency during evaluation
        self.check_inconsistencies = check_inconsistencies

        # this defines if NaN values should be replaced or not
        self.replace_nan_values_by = replace_nan_values_by

        # attribute which are excluded from being serialized )
        self.exclude_from_serialization = exclude_from_serialization if exclude_from_serialization is not None else []

        # the pareto set and front will be calculated only once and is stored here
        self._pareto_front = None
        self._pareto_set = None
        self._ideal_point, self._nadir_point = None, None

    def evaluate(self,
                 X,
                 *args,
                 return_values_of=None,
                 return_as_dictionary=False,
                 **kwargs):

        # make sure the array is at least 2d. store if reshaping was necessary
        X, only_single_value = at_least_2d_array(X, extend_as="row", return_if_reshaped=True)
        assert X.shape[1] == self.n_var, f'Input dimension {X.shape[1]} are not equal to n_var {self.n_var}!'

        # number of function evaluations to be done
        n_evals = X.shape[0]

        # the values to be actually returned by in the end - set bu default if not providded
        ret_vals = default_return_values(self.has_constraints()) if return_values_of is None else return_values_of

        # prepare the dictionary to be filled after the evaluation
        out = dict_with_none(ret_vals)

        # do the actual evaluation for the given problem - calls in _evaluate method internally
        self.do(X, out, *args, **kwargs)

        # make sure the array is 2d before doing the shape check
        out_to_2d_ndarray(out)

        # if enabled (recommended) the output shapes are checked for inconsistencies
        if self.check_inconsistencies:
            check(self, X, out)

        # if the NaN values should be replaced
        if self.replace_nan_values_by is not None:
            replace_nan_values(out, self.replace_nan_values_by)

        if "CV" in ret_vals or "feasible" in ret_vals:
            CV = calc_constr(out["G"]) if self.has_constraints() else np.zeros([n_evals, 1])
            out["CV"] = CV
            out["feasible"] = CV <= 0

        # in case the input had only one dimension, then remove always the first dimension from each output
        if only_single_value:
            out_to_1d_ndarray(out)

        # now depending on what should be returned prepare the output
        if return_as_dictionary:
            return out
        else:
            if len(ret_vals) == 1:
                return out[ret_vals[0]]
            else:
                return tuple([out[e] for e in ret_vals])

    def do(self, X, out, *args, **kwargs):
        self._evaluate(X, out, *args, **kwargs)
        out_to_2d_ndarray(out)

    def nadir_point(self):
        """
        Returns
        -------
        nadir_point : np.array
            The nadir point for a multi-objective problem. If single-objective, it returns the best possible solution
            which is equal to the ideal point.

        """
        # if the ideal point has not been calculated yet
        if self._nadir_point is None:

            # calculate the pareto front if not happened yet
            if self._pareto_front is None:
                self.pareto_front()

            # if already done or it was successful - calculate the ideal point
            if self._pareto_front is not None:
                self._nadir_point = np.max(self._pareto_front, axis=0)

        return self._nadir_point

    def ideal_point(self):
        """
        Returns
        -------
        ideal_point : np.array
            The ideal point for a multi-objective problem. If single-objective it returns the best possible solution.
        """

        # if the ideal point has not been calculated yet
        if self._ideal_point is None:

            # calculate the pareto front if not happened yet
            if self._pareto_front is None:
                self.pareto_front()

            # if already done or it was successful - calculate the ideal point
            if self._pareto_front is not None:
                self._ideal_point = np.min(self._pareto_front, axis=0)

        return self._ideal_point

    def pareto_front(self, *args, use_cache=True, exception_if_failing=False, **kwargs):
        """
        Parameters
        ----------

        args : Same problem implementation need some more information to create the Pareto front. For instance
                the DTLZ problem suite generates the Pareto front by usage of the reference directions.
                We refer to the corresponding problem for more information.
        exception_if_failing : bool
                Whether to throw an exception when generating the Pareto front has failed.
        use_cache : bool
                Whether to use the cache if the Pareto front has been generated beforehand.

        Returns
        -------
        P : np.array
            The Pareto front of a given problem. It is only loaded or calculate the first time and then cached.
            For a single-objective problem only one point is returned but still in a two dimensional array.

        """
        if not use_cache or self._pareto_front is None:
            try:
                pf = self._calc_pareto_front(*args, **kwargs)
                if pf is not None:
                    pf = at_least_2d_array(pf)

                self._pareto_front = pf
            except Exception as e:
                if exception_if_failing:
                    raise e

        return self._pareto_front

    def pareto_set(self, *args, use_cache=True, exception_if_failing=False, **kwargs):
        """
        Returns
        -------
        S : np.array
            Returns the pareto set for a problem. Points in the X space to be known to be optimal!
        """
        if not use_cache or self._pareto_set is None:
            try:
                ps = self._calc_pareto_set(*args, **kwargs)
                if ps is not None:
                    ps = at_least_2d_array(ps)
                self._pareto_set = ps
            except Exception as e:
                if exception_if_failing:
                    raise e

        return self._pareto_set

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
        """
        Method that either loads or calculates the pareto front. This is only done
        ones and the pareto front is stored.

        Returns
        -------
        pf : np.ndarray
            Pareto front as array.

        """
        pass

    def _calc_pareto_set(self, *args, **kwargs):
        pass

    @staticmethod
    def calc_constraint_violation(G):
        return calc_constr(G)

    def __str__(self):
        s = "# name: %s\n" % self.name()
        s += "# n_var: %s\n" % self.n_var
        s += "# n_obj: %s\n" % self.n_obj
        s += "# n_constr: %s\n" % self.n_constr
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


# ---------------------------------------------------------------------------------------------------------
# Elementwise Problem
# ---------------------------------------------------------------------------------------------------------


def elementwise_eval(problem, x, out, args, kwargs):
    problem._evaluate(x, out, *args, **kwargs)
    out_to_ndarray(out)
    check(problem, x, out)
    return out


def looped_eval(func_elementwise_eval, problem, X, out, *args, **kwargs):
    return [func_elementwise_eval(problem, x, dict(out), args, kwargs) for x in X]


def starmap_parallelized_eval(func_elementwise_eval, problem, X, out, *args, **kwargs):
    starmap = problem.starmap
    params = [(problem, x, dict(out), args, kwargs) for x in X]
    return list(starmap(func_elementwise_eval, params))


def dask_parallelized_eval(func_elementwise_eval, problem, X, out, *args, **kwargs):
    client = problem.client
    jobs = [client.submit(func_elementwise_eval, (problem, x, dict(out), args, kwargs)) for x in X]
    return [job.result() for job in jobs]


class ElementwiseProblem(Problem):

    def __init__(self,
                 func_elementwise_eval=elementwise_eval,
                 func_eval=looped_eval,
                 starmap=None,
                 exclude_from_serialization=None,
                 dask=None,
                 **kwargs):

        super().__init__(exclude_from_serialization=exclude_from_serialization, **kwargs)
        self.func_elementwise_eval = func_elementwise_eval
        self.func_eval = func_eval if starmap is None else starmap_parallelized_eval

        # the two ways of parallelization which are supported
        self.starmap = starmap
        self.dask = dask

        # do not serialize the starmap - this will throw an exception
        self.exclude_from_serialization = self.exclude_from_serialization + ["starmap", "dask"]

    def do(self, X, out, *args, **kwargs):

        # do an elementwise evaluation and return the results
        ret = self.func_eval(self.func_elementwise_eval, self, X, out, *args, **kwargs)

        # the first element decides what keys will be set
        keys = list(ret[0].keys())

        # now stack all the results for each of them together
        for key in keys:
            assert all([key in _out for _out in ret]), f"For some elements the {key} value has not been set."

            vals = []
            for elem in ret:
                val = elem[key]

                if val is not None:

                    # if it is just a float
                    if not isinstance(val, np.ndarray):
                        val = np.full((1, 1), val)
                    # otherwise prepare the value to be stacked with each other by extending the dimension
                    else:
                        val = val[None, ...]

                vals.append(val)

            # that means the key has never been set at all
            if all([val is None for val in vals]):
                out[key] = None
            else:
                out[key] = np.row_stack(vals)

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


def calc_constr(G):
    if G is None:
        return None
    elif G.ndim == 1 or G.shape[1] == 0:
        return np.zeros(len(G))[:, None]
    else:
        return np.maximum(0, G).sum(axis=1)[:, None]


def replace_nan_values(out, by=np.inf):
    for key in out:
        try:
            v = out[key]
            v[np.isnan(v)] = by
            out[key] = v
        except:
            pass


def check(problem, X, out):
    elementwise = X.ndim == 1

    # only used if not elementwise
    n_evals = X.shape[0]

    # the values from the output to be checked
    F, dF, G, dG = out.get("F"), out.get("dF"), out.get("G"), out.get("dG")

    if F is not None:
        correct = tuple([problem.n_obj]) if elementwise else (n_evals, problem.n_obj)
        assert F.shape == correct, f"Incorrect shape of F: {F.shape} != {correct} (provided != expected)"

    if dF is not None:
        correct = (problem.n_obj, problem.n_var) if elementwise else (n_evals, problem.n_obj, problem.n_var)
        assert dF.shape == correct, f"Incorrect shape of dF: {dF.shape} != {correct} (provided != expected)"

    if G is not None:
        if problem.has_constraints():
            correct = tuple([problem.n_constr]) if elementwise else (n_evals, problem.n_constr)
            assert G.shape == correct, f"Incorrect shape of G: {G.shape} != {correct} (provided != expected)"

    if dG is not None:
        if problem.has_constraints():
            correct = (problem.n_constr, problem.n_var) if elementwise else (n_evals, problem.n_constr, problem.n_var)
            assert dG.shape == correct, f"Incorrect shape of dG: {dG.shape} != {correct} (provided != expected)"
