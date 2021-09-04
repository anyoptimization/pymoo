import numpy as np

from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.solution import Solution, SolutionSet
from pymoo.problems.meta import MetaProblem


class NumericalDifferentiation(MetaProblem):

    def __init__(self, problem):
        super().__init__(problem)

    def do(self, X, out, *args, **kwargs):
        self.problem.do(X, out, *args, **kwargs)

        if "dF" in out:
            out["dF"] = NumericalDifferentiationUtil().do(self.problem, X, out["F"], return_values_of=["dF"])


EPS = np.finfo(float).eps


class NumericalDifferentiationUtil:

    def __init__(self, eps=None, jacobian=None, hessian=None):
        """

        eps as in https://stackoverflow.com/questions/31206443/numpy-second-derivative-of-a-ndimensional-array

        Parameters
        ----------
        """
        super().__init__()
        self.eps = eps

        self.jacobian = jacobian
        self.hessian = hessian

        if self.jacobian is None:
            self.jacobian = CentralJacobian()

        if self.hessian is None:
            self.hessian = CentralHessian()

    def do(self, problem, X, F=None, G=None, evaluator=None, hessian=False, return_values_of="auto"):

        # if no evaluator is provided to count function evaluations just use a plain one
        if evaluator is None:
            evaluator = Evaluator()

        # if it is not a population object yet, make one out of it
        pop = to_solution_set(X, F, G)

        # loop over each solution the approximation should be done for
        for solution in pop:
            x = solution.X

            # make sure the solution is evaluated
            if solution.F is None:
                evaluator.eval(problem, solution)

            eps = self.eps
            if eps is None:
                eps = EPS ** (1 / self.jacobian.n_points)
                eps = eps * np.maximum(eps, np.abs(x))

            if not isinstance(eps, np.ndarray):
                eps = np.full(len(x), eps)

            values = ["F"]
            if problem.has_constraints():
                values.append("G")

            self.calc(problem, evaluator, solution, values, eps, hessian)

        if return_values_of == "auto":
            return_values_of = ["dF"]
            if hessian:
                return_values_of.append("ddF")
            if problem.has_constraints():
                return_values_of.append("dG")
                if hessian:
                    return_values_of.append("ddG")

        if isinstance(X, Individual):
            pop = pop[0]

        ret = tuple([pop.get(e) for e in return_values_of])

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def calc(self, problem, evaluator, solution, values, eps, hessian):
        jacobian_estm, hessian_estm = self.jacobian, self.hessian

        jac_approx = Population.new(X=jacobian_estm.points(solution.X, eps))
        evaluator.eval(problem, jac_approx)

        for value in values:
            f, F = solution.get(value), jac_approx.get(value)
            dF = np.array([jacobian_estm.calc(f[m], F[:, m], eps) for m in range(F.shape[1])])
            solution.set("d" + value, dF)

        # if the hessian should be calculated as well
        if hessian:
            hess_approx = Population.new(X=self.hessian.points(solution.X, eps))
            evaluator.eval(problem, hess_approx)

            for value in values:
                f, F, FF = solution.get(value), jac_approx.get(value), hess_approx.get(value)
                ddF = np.array([hessian_estm.calc(f[m], F[:, m], FF[:, m], eps) for m in range(FF.shape[1])])
                solution.set("dd" + value, ddF)


# ---------------------------------------------------------------------------------------------------------
# Numerical Differentiation Estimators
# ---------------------------------------------------------------------------------------------------------


class NumericalDifferentiationEstimator:

    def points(self):
        pass

    def calc(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------------------------------------
# One Sided Differentiation Estimator
# ---------------------------------------------------------------------------------------------------------


class OneSidedJacobian(NumericalDifferentiationEstimator):
    n_points = 1

    def points(self, x, h):
        return x[None, :].repeat(len(x), axis=0) + np.eye(len(x)) * h

    def calc(self, f, f_jac, h):
        return (f_jac - f) / h


class OneSidedHessian(NumericalDifferentiationEstimator):
    n_points = 1

    def points(self, x, h):
        n = len(x)
        i, j = np.triu_indices(n)
        k = np.arange(len(i))

        eps = np.zeros((len(i), n))
        eps[k, i] += h[i]
        eps[k, j] += h[j]

        return x + eps

    def calc(self, f, f_jac, f_hess, h):
        assert f_jac.ndim == 1 and f_hess.ndim == 1 and len(f_jac) == int(np.sqrt(2 * (len(f_hess))))

        n = len(f_jac)
        i, j = np.triu_indices(n)

        deriv = (f_hess - f_jac[i] - f_jac[j] + f) / (h[i] * h[j])
        hessian = upper_diag_to_sym_full(deriv)

        return hessian


# ---------------------------------------------------------------------------------------------------------
# Central Differentiation Estimator
# ---------------------------------------------------------------------------------------------------------


class CentralJacobian(NumericalDifferentiationEstimator):
    n_points = 2

    def points(self, x, h):
        n = len(x)
        return x + np.row_stack([+1 * np.eye(n) * h, -1 * np.eye(n) * h])

    def calc(self, f, f_jac, h):
        n = int(len(f_jac) / 2)
        return (f_jac[:n] - f_jac[n:]) / (2 * h)


class CentralHessian(NumericalDifferentiationEstimator):
    n_points = 2

    def points(self, x, h):
        n = len(x)

        # [++, --] both having exactly n elements
        eps = [+1 * 2 * np.eye(n) * h, -1 * 2 * np.eye(n) * h]

        i, j = np.triu_indices(n, +1)
        k = np.arange(len(i))

        for s_1 in [+1, -1]:
            for s_2 in [+1, -1]:
                e = np.zeros((len(i), n))
                e[k, i] += s_1 * h[i]
                e[k, j] += s_2 * h[j]
                eps.append(e)

        eps = np.row_stack(eps)
        return x + eps

    def calc(self, f, f_jac, f_hess, h):
        n = int(len(f_jac) / 2)
        i, j = np.triu_indices(n, +1)
        d = np.arange(n)

        pos, neg, mixed = f_hess[:n], f_hess[n:2 * n], f_hess[2 * n:].reshape(4, -1)
        plus_plus, plus_minus, minus_plus, minus_minus = mixed

        hessian = np.zeros((n, n))
        hessian[d, d] = (- pos + 16 * f_jac[:n] - 30 * f + 16 * f_jac[n:] - neg) / (12 * h[d] ** 2)
        hessian[i, j] = (plus_plus - plus_minus - minus_plus + minus_minus) / (4 * h[i] * h[j])
        hessian[np.tril_indices(n, -1)] = hessian[np.triu_indices(n, +1)]

        return hessian


# ---------------------------------------------------------------------------------------------------------
# Complex Step Differentiation Estimator
#
# The Complex-Step Derivative Approximation' by Joaquim R. R. A. MARTINS, Peter STURDZA and Juan J. Alonso published in 2003.
# ---------------------------------------------------------------------------------------------------------


class ComplexStepJacobian(NumericalDifferentiationEstimator):
    n_points = 1

    def points(self, x, h):
        n = len(x)
        return x[None, :].repeat(n, axis=0) + 1j * np.eye(n) * h

    def calc(self, f, f_jac, h):
        return np.imag(f_jac) / h


# ---------------------------------------------------------------------------------------------------------
# Util Methods
# ---------------------------------------------------------------------------------------------------------

def to_solution_set(X, F=None, G=None):
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = Solution(X=X, F=F, G=G)
        else:
            X = Population.new(X=X, F=F, G=G)

    if isinstance(X, Individual):
        X = SolutionSet.create(X)

    return X


def upper_diag_to_sym_full(x):
    n = int(np.sqrt(2 * (len(x))))
    M = np.zeros((n, n))
    M[np.triu_indices(n)] = x
    M[np.tril_indices(n, -1)] = M[np.triu_indices(n, +1)]
    return M
