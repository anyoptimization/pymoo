import numpy as np

from pymoo.util.non_dominated_sorting import NonDominatedSorting


def get_extreme_points_c(F, ideal_point, extreme_points=None):
    # calculate the asf which is used for the extreme point decomposition
    asf = np.eye(F.shape[1])
    asf[asf == 0] = 1e-6

    # add the old extreme points to never loose them for normalization
    _F = F
    if extreme_points is not None:
        _F = np.concatenate([_F, extreme_points], axis=0)

    # use __F because we substitute small values to be 0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F / asf[:, None, :], axis=2)
    I = np.argmin(F_asf, axis=1)
    return I



if __name__ == '__main__':



    F = np.array([
        [1.0, 0.2, 0.0],
        [0.4, 0.1, 0.4],
        [0.1, 0.0, 1.0],
    ])



    plane = np.linalg.solve(F, np.ones(F.shape[1]))

    print(NonDominatedSorting().do(F))
    print(get_extreme_points_c(F, np.zeros(F.shape[1])))
    print(1/plane)

