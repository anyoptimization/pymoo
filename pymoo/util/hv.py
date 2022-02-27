import numpy as np

from pymoo.indicators.hv import Hypervolume
from pymoo.vendor.hv import HyperVolume


def hv(F, ref_point):
    hv = HyperVolume(ref_point)
    return hv.compute(F)


def calc_hvc_2d_slow(F, ref_point):
    n = len(F)

    I = np.lexsort((-F[:, 1], F[:, 0]))

    V = np.row_stack([ref_point, F[I], ref_point])

    hvi = np.zeros(n)

    for k in range(1, n + 1):
        height = V[k - 1, 1] - V[k, 1]
        width = V[k + 1, 0] - V[k, 0]

        hvi[I[k - 1]] = width * height

    return np.array(hvi)


def calc_hvc_2d_fast(F, ref_point):
    I = np.lexsort((-F[:, 1], F[:, 0]))
    V = np.row_stack([ref_point, F[I], ref_point])
    height = (V[:-1, 1] - V[1:, 1])[:-1]
    width = (V[1:, 0] - V[:-1, 0])[1:]
    return height * width


def calc_hvc_looped(F, ref_point, func=None):

    if func is None:
        hvc = Hypervolume(ref_point=ref_point, nds=False)
        func = lambda f, r: hvc.do(f)

    hv = func(F, ref_point)

    hvi = []

    for k in range(len(F)):
        v = np.full(len(F), True)
        v[k] = False

        _hv = func(F[v], ref_point)

        hvi.append(hv - _hv)

    hvi = np.array(hvi)

    return hvi

