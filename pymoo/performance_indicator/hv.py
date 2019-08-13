import os
import warnings

import numpy as np

from pymoo.model.indicator import Indicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize
from pymoo.vendor.hv import HyperVolume as _HyperVolume


def hypervolume_by_command(path_to_hv, X, ref_point):
    """
    A method to manually call the Hypervolume calculation if it is installed.
    http://lopez-ibanez.eu/hypervolume


    Parameters
    ----------
    path_to_hv : Path to the compiled executable
    X : Points to calculate the Hypervolume
    ref_point : Reference Point

    """

    ref_point_as_str = " ".join(format(x, ".3f") for x in ref_point)

    current_folder = os.path.dirname(os.path.abspath(__file__))

    path_to_input = os.path.join(current_folder, "in.dat")
    np.savetxt(path_to_input, X)

    path_to_output = os.path.join(current_folder, "out.dat")

    command = "%s -r \"%s\" %s > %s" % (path_to_hv, ref_point_as_str, path_to_input, path_to_output)
    #print(command)
    os.system(command)

    with open(path_to_output, 'r') as f:
        val = f.read()

    os.remove(path_to_input)
    os.remove(path_to_output)

    try:
        hv = float(val)
    except:
        warnings.warn(val)
        return - np.inf

    return hv


class Hypervolume(Indicator):

    def _calc(self, F):

        # only consider the non-dominated solutions for HV
        non_dom = NonDominatedSorting().do(F, only_non_dominated_front=True)
        _F = np.copy(F[non_dom, :])

        if self.normalize:
            # because we normalize now the reference point is (1,...1)
            ref_point = np.ones(F.shape[1])
            hv = _HyperVolume(ref_point)
            _F = normalize(_F, x_min=self.ideal_point, x_max=self.nadir_point)
        else:
            hv = _HyperVolume(self.ref_point)

        val = hv.compute(_F)
        return val
