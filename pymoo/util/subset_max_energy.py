import numpy as np

from pymoo.util.misc import vectorized_cdist


def squared_dist(A, B):
    return ((A[:, None] - B[None, :]) ** 2).sum(axis=2)


def calc_potential_energy(A):
    d = (A.shape[1] ** 2)
    i, j = np.triu_indices(len(A), 1)
    D = np.sqrt(squared_dist(A, A)[i, j])
    energy = np.log((1 / D ** d).mean())
    return energy


def subset_max_energy(X, n_survive):
    A = X

    def energy_if_replaced(A, B, i, j):
        _A = np.copy(A)
        _A[i] = B[j]
        return calc_potential_energy(_A)

    P = np.random.permutation(len(X))

    C = X[P[n_survive:]]
    X = X[P[:n_survive]]

    D_cand = np.sqrt(squared_dist(C, X))
    closest_to = D_cand.argmin(axis=1)

    has_improved = True
    energy = calc_potential_energy(X)

    while has_improved:

        has_improved = False

        for j in np.random.permutation(n_survive):
            I = np.where(closest_to == j)[0]

            if len(I) == 0:
                continue

            vals = np.array([energy_if_replaced(X, C, j, i) for i in I])

            _energy, i = vals.min(), I[vals.argmin()]

            if _energy < energy:
                point = X[j]
                X[j] = C[i]
                C[i] = point

                D_cand = np.sqrt(squared_dist(C, X))
                closest_to = D_cand.argmin(axis=1)

                energy = _energy
                has_improved = True

    return vectorized_cdist(X, A).argmin(axis=1)



