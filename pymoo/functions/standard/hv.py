"""
Standard Python implementation of hypervolume calculation.
"""

import numpy as np

from pymoo.vendor.hv import HyperVolume


def hv(ref_point, F):
    """Calculate hypervolume."""
    hv_calc = HyperVolume(ref_point)
    return hv_calc.compute(F)