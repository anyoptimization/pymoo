import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.variable import Real, get
from pymoo.operators.repair.bounds_repair import repair_random_init
from pymoo.util import default_random_state


@default_random_state
def pcx(X, eta, zeta, index, random_state=None):
    eps = 1e-32

    # the number of parents to be considered
    n_parents, n_matings, n_var = X.shape

    # calculate the differences from all parents to index parent
    diff_to_index = X - X[index]
    dist_to_index = np.linalg.norm(diff_to_index, axis=-1)
    dist_to_index = np.maximum(eps, dist_to_index)

    # find the centroid of the parents
    centroid = np.mean(X, axis=0)

    # calculate the difference between the centroid and the k-th parent
    diff_to_centroid = centroid - X[index]

    dist_to_centroid = np.linalg.norm(diff_to_centroid, axis=-1)
    dist_to_centroid = np.maximum(eps, dist_to_centroid)

    # orthogonal directions are computed
    orth_dir = np.zeros_like(dist_to_index)

    for i in range(n_parents):
        if i != index:
            temp1 = (diff_to_index[i] * diff_to_centroid).sum(axis=-1)
            temp2 = temp1 / (dist_to_index[i] * dist_to_centroid)
            temp3 = np.maximum(0.0, 1.0 - temp2 ** 2)
            orth_dir[i] = dist_to_index[i] * (temp3 ** 0.5)

    # this is the avg of the perpendicular distances from other parents to the parent k
    D_not = orth_dir.sum(axis=0) / (n_parents - 1)

    # generating zero-mean normally distributed variables
    sigma = D_not[:, None] * eta.repeat(n_var, axis=1)
    rnd = random_state.normal(loc=0.0, scale=sigma)

    # implemented just like the c code - generate_new.h file
    inner_prod = np.sum(rnd * diff_to_centroid, axis=-1, keepdims=True)
    noise = rnd - (inner_prod * diff_to_centroid) / dist_to_centroid[:, None] ** 2

    bias_to_centroid = random_state.normal(0.0, zeta) * diff_to_centroid

    # the array which is finally returned
    Xp = X[index] + noise + bias_to_centroid

    return Xp


class ParentCentricCrossover(Crossover):
    def __init__(self,
                 eta=0.1,
                 zeta=0.1,
                 **kwargs):

        super().__init__(n_parents=3, n_offsprings=1, **kwargs)
        self.eta = Real(eta, bounds=(0.01, 0.3))
        self.zeta = Real(zeta, bounds=(0.01, 0.3))

    def _do(self, problem, X, params=None, random_state=None, **kwargs):
        n_parents, n_matings, n_var = X.shape
        zeta, eta = get(self.zeta, self.eta, size=(n_matings, 1))

        index = 0

        Xp = pcx(X, eta, zeta, index=index, random_state=random_state)

        if problem.has_bounds():
            Xp = repair_random_init(Xp, X[index], *problem.bounds(), random_state=random_state)

        return Xp[None, :]


class PCX(ParentCentricCrossover):
    pass
