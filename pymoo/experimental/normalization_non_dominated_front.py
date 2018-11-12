import numpy as np


def get_nadir_point_from_fronts(F, fronts, ideal_point, epsilon=10e-3):
    n_obj = F.shape[1]
    nadir_point = np.zeros(n_obj)

    for m in range(n_obj):
        for k in range(len(fronts)):
            nadir_point[m] = np.max(F[fronts[k], m])
            if nadir_point[m] - ideal_point[m] > epsilon:
                break

    return nadir_point
