import numpy as np

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
    extreme_points = _F[I, :]

    return extreme_points


if __name__ == '__main__':


    I = np.zeros(3)

    F = np.array([
        [1.0, 0.0, 0.0],
        [0, 1e-7, 0],
        [0, 0, 1.0],
    ])

    print(get_extreme_points_c(F, I, None ))