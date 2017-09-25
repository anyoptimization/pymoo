from pygmo.core import hypervolume
import numpy as np

def calc_hypervolume(f, r):
    f = np.array([e for e in f if np.all(e < r)])
    if len(f) == 0:
        return 0.0
    hv = hypervolume(f)
    return hv.compute(r)
