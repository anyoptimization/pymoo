import numpy as np

from moocore import hypervolume as _hypervolume


def hv(ref_point, F):
    return _hypervolume(F, ref = ref_point)

def calc_hvc_looped(ref_point, F, func=hv):
    hv = func(F, ref_point)

    hvi = []

    for k in range(len(F)):
        v = np.full(len(F), True)
        v[k] = False
        _hv = func(F[v], ref_point)
        hvi.append(hv - _hv)

    hvi = np.array(hvi)
    return hvi
