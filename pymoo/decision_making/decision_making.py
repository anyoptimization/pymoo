import numpy as np


class DecisionMaking:

    def __init__(self, normalize=True, ideal_point=None, nadir_point=None) -> None:
        super().__init__()
        self.normalize = normalize
        self.ideal_point, self.nadir_point = ideal_point, nadir_point

    def do(self, F, *args, **kwargs):
        return self._do(F, *args, **kwargs)


def normalize(F, ideal_point=None, nadir_point=None, estimate_bounds_if_none=True, return_bounds=False):
    N = np.copy(F)

    if estimate_bounds_if_none:
        if ideal_point is None:
            ideal_point = np.min(F, axis=0)
        if nadir_point is None:
            nadir_point = np.max(F, axis=0)

    if ideal_point is not None:
        N -= ideal_point

    if nadir_point is not None:

        # calculate the norm for each objective
        norm = nadir_point - ideal_point

        # check if normalization makes sense
        if np.any(norm < 1e-8):
            raise Exception("Normalization failed because the range between the ideal and nadir point is not "
                            "large enough.")

        N /= norm

    else:
        norm = np.ones(F.shape[1])

    if return_bounds:
        return N, norm, ideal_point, nadir_point
    else:
        return N
