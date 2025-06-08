from pymoo.vendor.hv import HyperVolume


def hv(ref_point, F):
    hv_calc = HyperVolume(ref_point)
    return hv_calc.compute(F)