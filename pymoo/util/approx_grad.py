import numpy as np

from pymoo.model.evaluator import Evaluator
from pymoo.model.individual import Individual
from pymoo.model.population import Population


class GradientApproximation:

    def __init__(self, problem, epsilon=None, evaluator=None) -> None:
        super().__init__()
        self.epsilon = epsilon
        if self.epsilon is None:
            self.epsilon = np.sqrt(np.finfo(float).eps)
        self.problem = problem
        self.evaluator = evaluator if evaluator is not None else Evaluator()

    def do(self, individual, **kwargs):
        prob = self.problem
        n_var, n_obj, n_constr = prob.n_var, prob.n_obj, prob.n_constr
        individual = Individual(X=individual.X)
        self.evaluator.eval(self.problem, Population.create(individual))

        dF = np.zeros((n_obj, n_var))
        dG = np.zeros((n_constr, n_var))

        for i in range(n_var):
            x = np.copy(individual.X)
            x[i] += self.epsilon

            eps_F, _, eps_G = self.evaluator.eval(prob, x)
            dF[:, i] = (eps_F - individual.F) / self.epsilon

            if n_constr > 0 and eps_G is not None:
                dG[:, i] = (eps_G - individual.G) / self.epsilon

        return dF, dG


def approx_jacobian(x, func, epsilon, *args):
    """
    Approximate the Jacobian matrix of a callable function.

    Parameters
    ----------
    x : array_like
        The state vector at which to compute the Jacobian matrix.
    func : callable f(x,*args)
        The vector-valued function.
    epsilon : float
        The perturbation used to determine the partial derivatives.
    args : sequence
        Additional arguments passed to func.

    Returns
    -------
    An array of dimensions ``(lenf, lenx)`` where ``lenf`` is the length
    of the outputs of `func`, and ``lenx`` is the number of elements in
    `x`.

    Notes
    -----
    The approximation is done using forward differences.

    """
