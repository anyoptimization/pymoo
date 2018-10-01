import numpy as np


def naive_estimate_nadir_point(F, ideal_point, fronts, epsilon=1e-3):

    nadir_point = np.copy(ideal_point)

    for i in range(len(fronts)):

        mask = (ideal_point <= nadir_point - epsilon)

        if np.any(np.logical_not(mask)):
            break
        else:
            points = F[fronts[i], :]
            nadir_point[mask] = np.max(points, axis=0)[:, mask]

    return nadir_point
