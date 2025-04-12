
from pymoo.core.termination import NoTermination
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.util.display.single import SingleObjectiveOutput

try:
    from scipy.optimize import minimize as scipy_minimize, NonlinearConstraint, LinearConstraint
except:
    raise Exception("Please install SciPy: pip install scipy")

import warnings

import numpy as np

from pymoo.algorithms.base.local import LocalSearch
from pymoo.core.individual import Individual, constr_to_cv
from pymoo.core.population import Population

from pymoo.termination.max_gen import MaximumGenerationTermination



# ---------------------------------------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------------------------------------


class Optimizer(LocalSearch):

    def __init__(self, method, with_bounds=False, with_constr=False, require_jac=False,
                 use_bounds=True, use_constr=True, estm_gradients=True, disp=False, show_warnings=False, **kwargs):

        super().__init__(output=SingleObjectiveOutput(), **kwargs)

        self.method, self.with_bounds, self.with_constr, self.require_jac = method, with_bounds, with_constr, require_jac
        self.show_warnings = show_warnings
        self.use_bounds = use_bounds
        self.use_constr = use_constr
        self.estm_gradients = estm_gradients

        self.options = {
            'maxiter': int(1e8),  # because of C code interfacing this can not be inf
            'disp': disp}

    def _setup(self, problem, **kwargs):
        if isinstance(self.termination, MaximumGenerationTermination):
            self.options["maxiter"] = self.termination.n_max_gen
        elif isinstance(self.termination, MaximumFunctionCallTermination):
            self.options["maxfev"] = self.termination.n_max_evals

        self.termination = NoTermination()
        self.return_least_infeasible = True

    def _advance(self, **kwargs):
        problem, evaluator = self.problem, self.evaluator

        # add the box constraints defined in the problem
        bounds = None
        if self.use_bounds:

            xl, xu = self.problem.bounds()
            if self.with_bounds:
                bounds = np.column_stack([xl, xu])
            else:
                if xl is not None or xu is not None:
                    raise Exception(f"Error: Boundary constraints can not be handled by {self.method}")

        # define the actual constraints if supported by the algorithm
        constr = []
        if self.use_constr:

            constr = [LinearConstraint(np.eye(self.problem.n_var), xl, xu)]

            if problem.has_constraints():

                if self.with_constr:
                    def fun_constr(x):
                        g = problem.evaluate(x, return_values_of=["G"])
                        cv = constr_to_cv(g)
                        return cv

                    non_lin_constr = NonlinearConstraint(fun_constr, -float("inf"), 0)

                    constr.append(non_lin_constr)

                else:
                    raise Exception(f"Error: Constraint handling is not supported by {self.method}")

        # the objective function to be optimized and add gradients if available
        if self.estm_gradients:
            jac = None

            def fun_obj(x):
                f = problem.evaluate(x, return_values_of=["F"])[0]
                evaluator.n_eval += 1
                return f

        else:
            jac = True

            def fun_obj(x):
                f, df = problem.evaluate(x, return_values_of=["F", "dF"])

                if df is None:
                    raise Exception("If the gradient shall not be estimate, please set out['dF'] in _evaluate. ")

                evaluator.n_eval += 1
                return f[0], df[0]

        # the arguments to be used
        kwargs = dict(args=(), method=self.method, bounds=bounds, constraints=constr, jac=jac, options=self.options)

        # the starting solution found by sampling beforehand
        x0 = self.opt[0].X

        # actually run the optimization
        if not self.show_warnings:
            warnings.simplefilter("ignore")

        res = scipy_minimize(fun_obj, x0, **kwargs)

        opt = Population.create(Individual(X=res.x))
        self.evaluator.eval(self.problem, opt, algorithm=self)

        self.pop, self.off = opt, opt

        self.termination.force_termination = True

        if hasattr("res", "nit"):
            self.n_gen = res.nit + 1


# ---------------------------------------------------------------------------------------------------------
# Object Oriented Interface
# ---------------------------------------------------------------------------------------------------------

# +++++++++++++++++++++++++++++++++++++++++
# UNCONSTRAINED
# +++++++++++++++++++++++++++++++++++++++++

class NelderMead(Optimizer):

    def __init__(self, **kwargs):
        super().__init__("Nelder-Mead", **kwargs)


class CG(Optimizer):

    def __init__(self, **kwargs):
        super().__init__("CG", require_jac=True, **kwargs)


class NewtonCG(Optimizer):

    def __init__(self, **kwargs):
        super().__init__("Newton-CG", require_jac=True, **kwargs)


class BFGS(Optimizer):

    def __init__(self, **kwargs):
        super().__init__("BFGS", **kwargs)


class Powell(Optimizer):

    def __init__(self, **kwargs):
        super().__init__("Powell", **kwargs)


class Dogleg(Optimizer):

    def __init__(self, **kwargs):
        super().__init__("dogleg", require_jac=True, **kwargs)


class TrustNCG(Optimizer):

    def __init__(self, **kwargs):
        super().__init__("trust-ncg", require_jac=True, **kwargs)


class TrustExact(Optimizer):

    def __init__(self, **kwargs):
        super().__init__("trust-exact", require_jac=True, **kwargs)


class TrustKrylov(Optimizer):

    def __init__(self, **kwargs):
        super().__init__("trust-krylov", require_jac=True, **kwargs)


# +++++++++++++++++++++++++++++++++++++++++
# BOX CONSTRAINS
# +++++++++++++++++++++++++++++++++++++++++


class LBFGSB(Optimizer):

    def __init__(self, **kwargs):
        super().__init__("L-BFGS-B", with_bounds=True, **kwargs)


class TNC(Optimizer):

    def __init__(self, **kwargs):
        super().__init__("TNC", with_bounds=True, **kwargs)


# +++++++++++++++++++++++++++++++++++++++++
# NON-LINEAR CONSTRAINTS
# +++++++++++++++++++++++++++++++++++++++++


class COBYLA(Optimizer):

    def __init__(self, **kwargs):
        super().__init__("COBYLA", with_bounds=False, with_constr=True, **kwargs)


class SLSQP(Optimizer):

    def __init__(self, **kwargs):
        super().__init__("SLSQP", with_bounds=True, with_constr=True, **kwargs)


class TrustConstr(Optimizer):

    def __init__(self, **kwargs):
        super().__init__("trust-constr", with_bounds=True, with_constr=True, **kwargs)
