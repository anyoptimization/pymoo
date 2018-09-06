import numpy as np

from pymoo.util.mathematics import Mathematics


class Decomposition:

    def do(self, F, **kwargs):
        return self._do(F, **kwargs)


class PenaltyBasedBoundaryInterception(Decomposition):

    def __init__(self, theta) -> None:
        super().__init__()
        self.theta = theta

    def _do(self, F, weights, ideal_point, **kwargs):

        n_points, n_weights = F.shape[0], weights.shape[0]

        if n_points == n_weights:
            pass
        elif n_points == 1 and n_weights > 1:
            F = np.repeat(F, n_weights, axis=0)
        elif n_points > 1 and n_weights == 1:
            weights = np.repeat(weights, n_points, axis=0)
        else:
            raise Exception("Either for each point a weight, one weight, or one objective value.")

        return self.python_pbi(F, weights, ideal_point, self.theta)
        # return cython_pbi(F, weights, ideal_point, self.theta)

    def python_pbi(self, F, weights, ideal_point, theta):
        d1 = np.linalg.norm((F - ideal_point) * weights, axis=1) / np.linalg.norm(weights, axis=1)
        d2 = np.linalg.norm(F - (ideal_point - d1[:, None] * weights), axis=1)
        return d1 + theta * d2

    def python_obi_jmetal(self, F, weights, ideal_point, theta):

        """
        double
        d1, d2, nl;
        double
        theta = 5.0;

        d1 = d2 = nl = 0.0;

        for (int i = 0; i < problem.getNumberOfObjectives(); i++) {
        d1 += (individual.getObjective(i) - idealPoint[i]) * lambda[i];
        nl += Math.pow( lambda[i], 2.0);
        }
        nl = Math.sqrt(nl);
        d1 = Math.abs(d1) / nl;

        for (int i = 0; i < problem.getNumberOfObjectives(); i++) {
        d2 += Math.pow((individual.getObjective(i) - idealPoint[i]) - d1 * ( lambda[i] / nl), 2.0);
        }
        d2 = Math.sqrt(d2);

        fitness = (d1 + theta * d2);
        """

        nl = weights / np.linalg.norm(weights, axis=1)[:, None]



        d1 = np.linalg.norm((F - ideal_point) * weights, axis=1)
        d2 = np.linalg.norm(F - (ideal_point - d1[:, None] * weights), axis=1)
        return d1 + theta * d2


class WeightedSum(Decomposition):

    def _do(self, F, weights, **kwargs):
        return np.sum(F * weights, axis=1)


class Tchebicheff(Decomposition):

    def _do(self, F, weights, ideal_point, **kwargs):
        v = np.abs((F - ideal_point - Mathematics.EPS) * weights)
        return np.max(v, axis=1)


class DoNotKnow(Decomposition):
    def _do(self, F, weights, ideal_point, **kwargs):
        alpha = 0.1
        v = (F - ideal_point) * (1 / F.shape[1] + alpha * weights)
        return np.max(v, axis=1)
