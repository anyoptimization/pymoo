import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.operators.repair.inverse_penalty import InversePenaltyOutOfBoundsRepair, inverse_penality


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
                 n_offsprings=1,
                 prob_on_edge=0.2,
                 impl="vectorized",
                 **kwargs):

        super().__init__(n_parents=3, n_offsprings=n_offsprings, **kwargs)
        self.eta = float(eta)
        self.zeta = float(zeta)
        self.impl = impl
        self.prob_on_edge = prob_on_edge

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


"""
OUTDATED - Does not consider eps and all points being very close


def pcx_vectorized(X, k, eta, zeta, eps=0):

    # the number of parents to be considered
    n_parents, n_matings, n_var = X.shape

    # get the index parent for each mating
    index = X[k, range(n_matings)]

    # find the centroid of the parents
    centroid = np.mean(X, axis=0)

    # calculate the difference between the centroid and the k-th parent
    diff_to_centroid = eps_if_less_than_eps(centroid - index, eps)
    dist_to_centroid = np.linalg.norm(diff_to_centroid, axis=-1)

    # calculate the differences from all parents to parent k
    diff = eps_if_less_than_eps(X - index, eps)
    dist = np.linalg.norm(diff, axis=-1)

    # orthogonal directions are computed
    orth_dir = np.zeros((n_matings, n_parents))

    for i in range(n_parents):
        temp1 = (diff[i] * diff_to_centroid).sum(axis=-1)
        temp2 = temp1 / (dist[i] * dist_to_centroid)
        temp3 = np.maximum(1.0 - temp2 ** 2, 0)
        orth_dir[:, i] = dist[i] * temp3 ** 0.5

    # this is the avg of the perpendicular distances from other parents to the parent k
    D_not = orth_dir.sum(axis=-1) / (n_parents - 1)

    # generating zero-mean normally distributed variables
    sigma = (D_not * eta)
    rnd = np.random.randn(n_matings, n_var) * sigma[:, None]

    # implemented just like the c code - generate_new.h file
    inner_prod = np.sum(rnd * diff_to_centroid, axis=-1)
    noise = rnd - (inner_prod[:, None] * diff_to_centroid) / dist_to_centroid[:, None] ** 2

    # calculate the bias towards the center
    bias_to_centroid = diff_to_centroid * np.random.randn(n_matings, 1) * zeta

    # create the final offsprings
    off = index + noise + bias_to_centroid

    return off[None, :]

"""
