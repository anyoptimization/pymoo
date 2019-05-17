import numpy as np

if __name__ == "__main__":

    fname = "c1dtlz3"

    n_obj = 3
    n_constr = 1
    n_var = 12

    M = np.loadtxt(fname + ".out")

    F = M[:, :n_obj]
    G = M[:, n_obj:n_obj+n_constr]
    X = M[:, n_obj+n_constr:n_obj+n_constr+n_var]
    CV = M[:, n_obj+n_constr+n_var:n_obj+n_constr+n_var+1]

    G = -G
    CV = -CV

    np.savetxt(fname + ".x", X)
    np.savetxt(fname + ".f", F)
    np.savetxt(fname + ".cv", CV)