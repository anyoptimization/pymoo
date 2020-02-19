import numpy as np

from pymoo.vendor.hv import HyperVolume

data = np.array([[0, 3, 0], [1, 2, 0], [2, 1, 0], [3, 0, 0]]).astype('double')
ref_point = np.array([3.0, 3.0, 0])
HyperVolume(ref_point).compute(data)
