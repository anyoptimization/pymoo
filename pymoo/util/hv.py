import numpy as np

from pymoo.vendor.hv import HyperVolume


def hv(ref_point, F):
    hv = HyperVolume(ref_point)
    return hv.compute(F)


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
