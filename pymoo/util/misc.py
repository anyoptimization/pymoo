import numpy as np




def denormalize(x, x_min, x_max):
    return x * (x_max - x_min) + x_min


def normalize(x, x_min=None, x_max=None, return_bounds=False):
    if x_min is None:
        x_min = np.min(x, axis=0)
    if x_max is None:
        x_max = np.max(x, axis=0)

    res = (x - x_min) / (x_max - x_min)
    if not return_bounds:
        return res
    else:
        return res, x_min, x_max


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

