import numpy as np

from pymoo.operators.repair.bounds_repair import BoundsRepair


def inverse_penality(x, p, xl, xu, alpha=None):
    assert len(p) == len(x)

    normv = np.linalg.norm(p - x)

    # violated boundaries
    idl = x < xl
    idr = xu < x

    # if nothing is out of bounds just return the value
    if not np.any(np.logical_or(idl, idr)):
        return x

    else:
        # lower bounds of Y
        diff = (p - x)
        diff[diff == 0] = 1e-32
        d = normv * np.max(np.maximum(idl * (xl - x) / diff, idr * (xu - x) / diff))

        # upper bounds on Y
        bounds = np.array([~idl * ((xl - x) / diff), ~idr * (xu - x) / diff])

        D = normv * np.min(bounds[bounds > 0])

        if alpha is None:
            alpha = (normv - d) / normv
            alpha += 1e-32

        r = np.random.random()
        Y = d * (1.0 + alpha * np.tan(r * np.arctan((D - d) / (alpha * d))))

        ret = x + (p - x) * Y / normv

        # for floating point error - in theory it will always be in bounds
        ret[ret < xl] = xl[ret < xl]
        ret[ret > xu] = xu[ret > xu]

        return ret


def inverse_penality_by_problem(problem, x, p, **kwargs):
    return inverse_penality(x, p, problem.xl, problem.xu, **kwargs)


class InversePenaltyOutOfBoundsRepair(BoundsRepair):

    def repair_out_of_bounds(self, problem, X, P=None, **kwargs):
        if P is None:
            raise Exception("For this out of bounds handling a parent solution in bounds needs to be provided.")
        assert len(X) == len(P)
        n = len(X)

        for k in range(n):
            X[k] = inverse_penality_by_problem(problem, X[k], P[k])

        return X


if __name__ == '__main__':

    # lower and upper bounds
    xl = np.zeros(2)
    xu = np.ones(2)

    # chosen parents
    # p = np.array([0.1, 1.0])
    p = np.array([0.5, 0.6])

    c = np.array([-0.1, 1.0])

    import matplotlib.pyplot as plt

    plt.scatter(p[0], p[1], color="green", label="Parent")
    plt.scatter(c[0], c[1], color="orange", label="Offspring")

    data = []
    for j in range(200):
        ret = inverse_penality(c, p, xl, xu, alpha=None)
        plt.scatter(ret[0], ret[1], facecolor="none", edgecolor="red", s=10, alpha=0.6)

    plt.ylim(0.0, 1.3)
    plt.xlim(-0.2, 1)
    plt.legend()
    plt.show()
