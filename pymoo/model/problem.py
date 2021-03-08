import multiprocessing
import warnings
from abc import abstractmethod
from multiprocessing.pool import ThreadPool

import autograd
import autograd.numpy as anp
import numpy as np

from pymoo.problems.gradient import run_and_trace, calc_jacobian
from pymoo.util.misc import at_least_2d_array


class Problem:
    """
    Superclass for each problem that is defined. It provides attributes such
    as the number of variables, number of objectives or constraints.
    Also, the lower and upper bounds are stored. If available the Pareto-front, nadir point
    and ideal point are stored.

    """

    def __init__(self,
                 n_var=-1,
                 n_obj=-1,
                 n_constr=0,
                 xl=None,
                 xu=None,
                 type_var=np.double,
                 evaluation_of="auto",
                 replace_nan_values_of="auto",
                 parallelization=None,
                 elementwise_evaluation=False,
                 exclude_from_serialization=["parallelization"],
                 callback=None):
        """

        Parameters
        ----------
        n_var : int
            number of variables
        n_obj : int
            number of objectives
        n_constr : int
            number of constraints
        xl : np.array or int
            lower bounds for the variables. if integer all lower bounds are equal.
        xu : np.array or int
            upper bounds for the variable. if integer all upper bounds are equal.
        type_var : numpy type
            type of the variable to be evaluated. Can also be np.object if it is a complex data type
        elementwise_evaluation : bool

        parallelization : str or tuple
            See :ref:`nb_parallelization` for guidance on parallelization.

        """

        # number of variable for this problem
        self.n_var = n_var

        # type of the variable to be evaluated
        self.type_var = type_var

        # number of objectives
        self.n_obj = n_obj

        # number of constraints
        self.n_constr = n_constr

        # whether box boundaries (xl, xu) should be handled as constraints during the optimization
        self.bounds_as_constraints = False

        # allow just an integer for xl and xu if all bounds are equal
        if n_var > 0 and not isinstance(xl, np.ndarray) and xl is not None:
            self.xl = np.ones(n_var) * xl
        else:
            self.xl = xl

        if n_var > 0 and not isinstance(xu, np.ndarray) and xu is not None:
            self.xu = np.ones(n_var) * xu
        else:
            self.xu = xu

        # the pareto set and front will be calculated only once and is stored here
        self._pareto_front = None
        self._pareto_set = None
        self._ideal_point, self._nadir_point = None, None

        # actually defines what _evaluate is setting during the evaluation
        if evaluation_of == "auto":
            # by default F is set, and G if the problem does have constraints
            self.evaluation_of = ["F"]
            if self.n_constr > 0:
                self.evaluation_of.append("G")
        else:
            self.evaluation_of = evaluation_of

        # if nan values should be replace
        if replace_nan_values_of == "auto":
            self.replace_nan_values_of = ["F", "G"] if self.has_constraints() else ["F"]
        else:
            self.replace_nan_values_of = replace_nan_values_of

        # whether the evaluation function is called per set of solutions or single solution
        self.elementwise_evaluation = elementwise_evaluation

        # only applicable if elementwise_evaluation is true - if, how should the single evaluations be parallelized
        self.parallelization = parallelization

        # attribute which are excluded from being serialized )
        self.exclude_from_serialization = exclude_from_serialization

        # store the callback if defined
        self.callback = callback

    def nadir_point(self):
        """
        Returns
        -------
        nadir_point : np.array
            The nadir point for a multi-objective problem.
            If single-objective, it returns the best possible solution which is equal to the ideal point.

        """
        # if the ideal point has not been calculated yet
        if self._nadir_point is None:

            # calculate the pareto front if not happened yet
            if self._pareto_front is None:
                self.pareto_front()

            # if already done or it was successful - calculate the ideal point
            if self._pareto_front is not None:
                self._ideal_point = np.max(self._pareto_front, axis=0)

        return self._nadir_point

    def ideal_point(self):
        """
        Returns
        -------
        ideal_point : np.array
            The ideal point for a multi-objective problem. If single-objective
            it returns the best possible solution.
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

    def pareto_front(self, *args, use_cache=True, exception_if_failing=True, **kwargs):
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

    def pareto_set(self, *args, use_cache=True, **kwargs):
        """
        Returns
        -------
        S : np.array
            Returns the pareto set for a problem. Points in the X space to be known to be optimal!
        """
        if not use_cache or self._pareto_set is None:
            ps = self._calc_pareto_set(*args, **kwargs)
            if ps is not None:
                ps = at_least_2d_array(ps)
            self._pareto_set = ps

        return self._pareto_set

    def evaluate(self,
                 X,
                 *args,
                 return_values_of="auto",
                 return_as_dictionary=False,
                 **kwargs):

        """
        Evaluate the given problem.

        The function values set as defined in the function.
        The constraint values are meant to be positive if infeasible. A higher positive values means "more" infeasible".
        If they are 0 or negative, they will be considered as feasible what ever their value is.

        Parameters
        ----------

        X : np.array
            A two dimensional matrix where each row is a point to evaluate and each column a variable.

        return_as_dictionary : bool
            If this is true than only one object, a dictionary, is returned. This contains all the results
            that are defined by return_values_of. Otherwise, by default a tuple as defined is returned.

        return_values_of : list of strings
            You can provide a list of strings which defines the values that are returned. By default it is set to
            "auto" which means depending on the problem the function values or additional the constraint violation (if
            the problem has constraints) are returned. Otherwise, you can provide a list of values to be returned.

            Allowed is ["F", "CV", "G", "dF", "dG", "dCV", "feasible"] where the d stands for
            derivative and h stands for hessian matrix.


        Returns
        -------

            A dictionary, if return_as_dictionary enabled, or a list of values as defined in return_values_of.

        """

        # call the callback of the problem
        if self.callback is not None:
            self.callback(X)

        # make the array at least 2-d - even if only one row should be evaluated
        only_single_value = len(np.shape(X)) == 1
        X = np.atleast_2d(X)

        # check the dimensionality of the problem and the given input
        if X.shape[1] != self.n_var:
            raise Exception('Input dimension %s are not equal to n_var %s!' % (X.shape[1], self.n_var))

        # automatic return the function values and CV if it has constraints if not defined otherwise
        if type(return_values_of) == str and return_values_of == "auto":
            return_values_of = ["F"]
            if self.n_constr > 0:
                return_values_of.append("CV")

        # all values that are set in the evaluation function
        values_not_set = [val for val in return_values_of if val not in self.evaluation_of]

        # have a look if gradients are not set and try to use autograd and calculate grading if implemented using it
        gradients_not_set = [val for val in values_not_set if val.startswith("d")]

        # whether gradient calculation is necessary or not
        calc_gradient = (len(gradients_not_set) > 0)

        # set in the dictionary if the output should be calculated - can be used for the gradient
        out = {}
        for val in return_values_of:
            out[val] = None

        # calculate the output array - either elementwise or not. also consider the gradient
        if self.elementwise_evaluation:
            out = self._evaluate_elementwise(X, calc_gradient, out, *args, **kwargs)
        else:
            out = self._evaluate_batch(X, calc_gradient, out, *args, **kwargs)

            calc_gradient_of = [key for key, val in out.items()
                                if "d" + key in return_values_of and
                                out.get("d" + key) is None and
                                (type(val) == autograd.numpy.numpy_boxes.ArrayBox)]

            if len(calc_gradient_of) > 0:
                deriv = self._calc_gradient(out, calc_gradient_of)
                out = {**out, **deriv}

        # convert back to conventional numpy arrays - no array box as return type
        for key in out.keys():
            if type(out[key]) == autograd.numpy.numpy_boxes.ArrayBox:
                out[key] = out[key]._value

        # add the boundary constraints if they are supposed to be added
        if self.bounds_as_constraints:

            # get the boundaries for normalization
            xl, xu = self.bounds()

            # add the boundary constraint if enabled
            _G = np.zeros((len(X), 2 * self.n_var))
            _G[:, :self.n_var] = (xl - X)
            _G[:, self.n_var:] = (X - xu)

            # attach the constraints to the results
            out["G"] = np.column_stack([out["G"], _G]) if out["G"] is not None else _G

            if "dG" in out:
                _dG = np.zeros((len(X), 2 * self.n_var, self.n_var))
                _dG[:, :self.n_var, :] = - np.eye(self.n_var)
                _dG[:, self.n_var:, :] = np.eye(self.n_var)

                out["dG"] = np.column_stack([out["dG"], _dG]) if out["dG"] is not None else _dG

        # replace non values by infinity - because of minimization it serves like a large penalty
        for key in self.replace_nan_values_of:
            if key in out:
                try:
                    v = out[key]
                    v[np.isnan(v)] = np.inf
                    out[key] = v
                except:
                    pass

        # if constraint violation should be returned as well
        if self.n_constr == 0:
            CV = np.zeros([X.shape[0], 1])
        else:
            CV = Problem.calc_constraint_violation(out["G"])

        if "CV" in return_values_of:
            out["CV"] = CV

        # if an additional boolean flag for feasibility should be returned
        if "feasible" in return_values_of:
            out["feasible"] = (CV <= 0)

        # if asked for a value but not set in the evaluation set to None
        for key in return_values_of:
            if key not in out:
                out[key] = None

        # remove the first dimension of the output - in case input was a 1d- vector
        if only_single_value:
            for key in out.keys():
                if out[key] is not None:
                    out[key] = out[key][0, :]

        if return_as_dictionary:
            return out
        else:

            # if just a single value do not return a tuple
            if len(return_values_of) == 1:
                return out[return_values_of[0]]
            else:
                return tuple([out[val] for val in return_values_of])

    def _calc_gradient(self, out, keys):

        deriv = {}
        for key in keys:
            val = out[key]

            # calculate the jacobian matrix and set it - (ignore warnings of autograd here)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jac = calc_jacobian(out["__autograd__"], val)
                deriv["d" + key] = jac

        return deriv

    def _evaluate_batch(self, X, calc_gradient, out, *args, **kwargs):
        if calc_gradient:
            out["__autograd__"], _ = run_and_trace(self._evaluate, X, *[out])
        else:
            self._evaluate(X, out, *args, **kwargs)
        at_least2d(out)

        return out

    def _evaluate_elementwise(self, X, calc_gradient, out, *args, **kwargs):
        ret = []

        def func(_x):
            _out = {}
            if calc_gradient:
                grad, _ = run_and_trace(self._evaluate, _x, *[_out])
                _out["__autograd__"] = grad
            else:
                self._evaluate(_x, _out, *args, **kwargs)
            return _out

        parallelization = self.parallelization
        if not isinstance(parallelization, (list, tuple)):
            parallelization = [self.parallelization]

        _type = parallelization[0]
        if len(parallelization) >= 1:
            _params = parallelization[1:]

        # just serialize evaluation
        if _type is None:
            [ret.append(func(x)) for x in X]

        elif _type == "starmap":
            if len(_params) != 1:
                raise Exception("The starmap parallelization method must be accompanied by a starmapping callable")

            params = [[X[k], calc_gradient, self._evaluate, args, kwargs] for k in range(len(X))]

            starmapper = _params[0]
            ret = list(starmapper(evaluate_in_parallel, params))

        elif _type == "threads":

            if len(_params) == 0:
                n_threads = multiprocessing.cpu_count() - 1
            else:
                n_threads = _params[0]

            with ThreadPool(n_threads) as pool:
                params = [[X[k], calc_gradient, self._evaluate, args, kwargs] for k in range(len(X))]
                ret = pool.starmap(evaluate_in_parallel, params)

        elif _type == "dask":

            if len(_params) != 2:
                raise Exception("A distributed client objective is need for using dask. parallelization=(dask, "
                                "<client>, <function>).")
            else:
                client, fun = _params

            jobs = []
            for k in range(len(X)):
                jobs.append(client.submit(fun, X[k]))

            ret = [job.result() for job in jobs]

        else:
            raise Exception(
                "Unknown parallelization method: %s (should be one of: None, starmap, threads, dask)" % _type)

        # stack all the single outputs together
        for key in ret[0].keys():
            out[key] = anp.row_stack([ret[i][key] for i in range(len(ret))])

        return out

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
        """
        Returns
        -------
        name : str
            The name of the problem. Per default it is the name of the class but it can be overridden.
        """
        return self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        """
        Method that either loads or calculates the pareto front. This is only done
        ones and the pareto front is stored.

        Returns
        -------
        pf : np.array
            Pareto front as array.

        """
        pass

    def _calc_pareto_set(self, *args, **kwargs):
        pass

    # some problem information
    def __str__(self):
        s = "# name: %s\n" % self.name()
        s += "# n_var: %s\n" % self.n_var
        s += "# n_obj: %s\n" % self.n_obj
        s += "# n_constr: %s\n" % self.n_constr
        return s

    @staticmethod
    def calc_constraint_violation(G):
        if G is None:
            return None
        elif G.shape[1] == 0:
            return np.zeros(G.shape[0])[:, None]
        else:
            return np.sum(G * (G > 0).astype(float), axis=1)[:, None]

    def __getstate__(self):
        state = self.__dict__.copy()
        # exclude objects which should not be stored
        for key in self.exclude_from_serialization:
            state[key] = None
        return state

    def set_boundaries_as_constraints(self, val=True):
        if self.bounds_as_constraints and not val:
            self.bounds_as_constraints = False
            self.n_constr -= 2 * self.n_var
        elif not self.bounds_as_constraints and val:
            self.bounds_as_constraints = True
            self.n_constr += 2 * self.n_var


# makes all the output at least 2-d dimensional
def at_least2d(d):
    for key in d.keys():
        if len(np.shape(d[key])) == 1:
            d[key] = d[key][:, None]


def evaluate_in_parallel(_x, calc_gradient, func, args, kwargs):
    _out = {}
    if calc_gradient:
        _out["__autograd__"], _ = run_and_trace(func, _x, *[_out])
    else:
        func(_x, _out, *args, **kwargs)
    return _out


def evaluate_in_parallel_object(_x, calc_gradient, obj, args, kwargs):
    _out = {}
    obj._evaluate(_x, _out, *args, **kwargs)
    return _out


def func_return_none(*args, **kwargs):
    return None


class FunctionalProblem(Problem):

    def __init__(self,
                 n_var,
                 objs,
                 constr_ieq=[],
                 constr_eq=[],
                 constr_eq_eps=1e-6,
                 func_pf=func_return_none,
                 func_ps=func_return_none,
                 **kwargs):
        if callable(objs):
            objs = [objs]

        self.objs = objs
        self.constr_ieq = constr_ieq
        self.constr_eq = constr_eq
        self.constr_eq_eps = constr_eq_eps
        self.func_pf = func_pf
        self.func_ps = func_ps

        n_constr = len(constr_ieq) + len(constr_eq)

        super().__init__(n_var,
                         n_obj=len(self.objs),
                         n_constr=n_constr,
                         elementwise_evaluation=True,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        # calculate violation from the inequality constraints
        ieq = np.array([constr(x) for constr in self.constr_ieq])
        ieq[ieq < 0] = 0

        # calculate violation from the quality constraints
        eq = np.array([constr(x) for constr in self.constr_eq])
        eq = np.abs(eq)
        eq = eq - self.constr_eq_eps

        # calculate the objective function
        f = np.array([obj(x) for obj in self.objs])

        out["F"] = f
        out["G"] = np.concatenate([ieq, eq])

    def _calc_pareto_front(self, *args, **kwargs):
        return self.func_pf(*args, **kwargs)

    def _calc_pareto_set(self, *args, **kwargs):
        return self.func_ps(*args, **kwargs)


class MetaProblem(Problem):

    def __init__(self, problem):
        super().__init__(n_var=problem.n_var,
                         n_obj=problem.n_obj,
                         n_constr=problem.n_constr,
                         xl=problem.xl,
                         xu=problem.xu,
                         type_var=problem.type_var,
                         evaluation_of=problem.evaluation_of,
                         parallelization=problem.parallelization,
                         elementwise_evaluation=problem.elementwise_evaluation,
                         callback=problem.callback)

        self.problem = problem

    def _evaluate(self, x, out, *args, **kwargs):
        self.problem._evaluate(x, out, *args, **kwargs)

    def pareto_front(self, *args, **kwargs):
        return self.problem.pareto_front(*args, **kwargs)

    def pareto_set(self, *args, **kwargs):
        return self.problem.pareto_set(*args, **kwargs)


class ConstraintsAsPenaltyProblem(MetaProblem):

    def __init__(self,
                 problem,
                 penalty=1e6):
        super().__init__(problem)
        self.penalty = penalty
        self.n_constr = 0

    def _evaluate(self, x, out, *args, **kwargs):
        kwargs["return_as_dictionary"] = True
        super()._evaluate(x, out, *args, **kwargs)

        F, G = at_least_2d_array(out["F"]), at_least_2d_array(out["G"])
        CV = Problem.calc_constraint_violation(G)

        out["__F__"] = F
        out["__G__"] = G
        out["__CV__"] = CV

        out["F"] = F + self.penalty * CV
        out["G"] = None

    def pareto_front(self, *args, **kwargs):
        return self.problem.pareto_front(*args, **kwargs)

    def pareto_set(self, *args, **kwargs):
        return self.problem.pareto_set(*args, **kwargs)


class StaticProblem(MetaProblem):

    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        self.kwargs = kwargs

    def _evaluate(self, x, out, *args, **kwargs):
        for K, V in self.kwargs.items():
            out[K] = V
