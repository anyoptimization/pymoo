import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.operators.repair.inverse_penalty import inverse_penality


def eps_if_less_than_eps(x, eps):
    x[x < eps] = eps
    return x


def pcx(X, k, eta, zeta):
    # the number of parents to be considered
    n_parents, n_var = X.shape

    # find the centroid of the parents
    centroid = np.mean(X, axis=0)

    # calculate the difference between the centroid and the k-th parent
    diff_to_centroid = centroid - X[k]
    dist_to_centroid = np.linalg.norm(diff_to_centroid)

    # calculate the differences from all parents to parent k
    diff_to_index = X - X[k]
    dist_to_index = np.linalg.norm(diff_to_index, axis=1)

    # orthogonal directions are computed
    orth_dir = np.zeros(n_parents)

    S = dist_to_index > 1e-16
    if S.sum() == 0:
        return centroid

    for i in range(n_parents):
        if S[i]:
            temp1 = (diff_to_index[i] * diff_to_centroid).sum()
            temp2 = temp1 / (dist_to_index[i] * dist_to_centroid)
            temp3 = max(1.0 - temp2 ** 2, 0)
            orth_dir[i] = dist_to_index[i] * temp3 ** 0.5

    # this is the avg of the perpendicular distances from other parents to the parent k
    D_not = orth_dir.sum() / (n_parents - 1)

    # generating zero-mean normally distributed variables
    mu, sigma = 0.0, (D_not * eta)
    rnd = np.random.normal(mu, sigma, n_var)

    # implemented just like the c code - generate_new.h file
    noise = rnd - (np.sum(rnd * diff_to_centroid) * diff_to_centroid) / dist_to_centroid ** 2

    bias_to_centroid = diff_to_centroid * np.random.normal(0.0, zeta, 1)

    off = X[k] + noise + bias_to_centroid

    return off


class ParentCentricCrossover(Crossover):
    def __init__(self,
                 eta=0.1,
                 zeta=0.1,
                 **kwargs):

        super().__init__(n_parents=3, n_offsprings=1, **kwargs)
        self.eta = float(eta)
        self.zeta = float(zeta)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        off = np.empty([self.n_offsprings, n_matings, n_var])
        K = np.row_stack([np.random.permutation(self.n_parents) for _ in range(n_matings)])[:, :self.n_offsprings]

        for j in range(self.n_offsprings):
            for i in range(n_matings):
                x = X[:, i, :]
                k = K[i, j]

                # do the crossover
                _off = pcx(x, k, self.eta, self.zeta)

                # make sure the offspring is in bounds and assign
                off[j, i, :] = inverse_penality(_off, x[k], problem.xl, problem.xu)

        return off


class PCX(ParentCentricCrossover):
    pass
