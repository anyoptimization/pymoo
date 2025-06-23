from moocore import hypervolume as _hypervolume


def hv(ref_point, F):
    return _hypervolume(F, ref = ref_point)
