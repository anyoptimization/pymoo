import numpy as np
from numpy.linalg import LinAlgError


def denormalize(x, x_min, x_max):
    return x * (x_max - x_min) + x_min

def normalize(x, x_min=None, x_max=None, return_bounds=False):
    if x_min is None:
        x_min = np.min(x, axis=0)
    if x_max is None:
        x_max = np.max(x, axis=0)

    denom = x_max - x_min
    denom += 1e-30

    res = (x - x_min) / denom
    if not return_bounds:
        return res
    else:
        return res, x_min, x_max



def normalize_by_asf_interceptions(x, return_bounds=False):

    # find the x_min point
    n_obj = x.shape[1]
    x_min = np.min(x, axis=0)
    # transform the objective that 0 means the best
    F = x - x_min

    # calculate the asf matrix
    asf = np.eye(n_obj)
    asf = asf + (asf == 0) * 1e-16

    # result matrix with the selected points
    S = np.zeros((n_obj, n_obj))

    # find for each direction the best
    for i in range(len(asf)):
        val = np.max(F / asf[i, :], axis=1) + 0.0001 * np.sum(F / asf[i, :])
        S[i, :] = F[np.argmin(val), :]

    try:
        b = np.ones(n_obj)
        A = np.linalg.solve(S, b)
        A[A==0] = 0.000001
        A = 1 / A
        A = A.T
    except LinAlgError:
        A = np.max(S, axis=0)

    F = F / A

    if not return_bounds:
        return F
    else:
        x_max = A + x_min
        return F, x_min, x_max


def calc_constraint_violation(G):
    return np.sum(G * (G > 0).astype(np.float), axis=1)


# returns only the unique rows from a given matrix X
def unique_rows(X):
    y = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
    _, idx = np.unique(y, return_index=True)
    return idx


def create_hist(n_evals, pop):
    return np.concatenate((np.ones((pop.size(), 1)) * n_evals, pop.X, pop.F, pop.G), axis=1)


def save_hist(pathToFile, data):
    hist = None
    for i in range(len(data)):
        obj = data[i]['snapshot']
        hist = obj if hist is None else np.concatenate((hist, obj), axis=0)

    np.savetxt(pathToFile, hist, fmt='%.14f')
    print(pathToFile)
