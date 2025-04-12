import numpy as np

from pymoo.indicators.hv.exact import ExactHypervolume


def hvc_2d_slow(ref_point, F):
    n = len(F)

    I = np.lexsort((-F[:, 1], F[:, 0]))

    V = np.row_stack([ref_point, F[I], ref_point])

    hvi = np.zeros(n)

    for k in range(1, n + 1):
        height = V[k - 1, 1] - V[k, 1]
        width = V[k + 1, 0] - V[k, 0]

        hvi[I[k - 1]] = width * height

    return np.array(hvi)


def hvc_2d_fast(ref_point, F_sorted, left=None, right=None):
    if left is None:
        left = [F_sorted[0, 0], ref_point[1]]

    if right is None:
        right = [ref_point[0], F_sorted[-1, 1]]

    V = np.row_stack([left, F_sorted, right])
    height = (V[:-1, 1] - V[1:, 1])[:-1]
    width = (V[1:, 0] - V[:-1, 0])[1:]

    hvc = height * width
    return hvc


def hv_2d_fast(ref_point, F_sorted):
    V = np.row_stack([ref_point, F_sorted])
    height = (V[:-1, 1] - V[1:, 1])
    width = ref_point[0] - V[1:, 0]
    return (height * width).sum()


class ExactHypervolume2D(ExactHypervolume):

    def __init__(self, ref_point, **kwargs) -> None:
        assert len(ref_point) == 2, "This hypervolume calculation only works in 2 dimensions."
        super().__init__(ref_point, func_hv=hv_2d_fast, func_hvc=hvc_2d_fast, **kwargs)
        self.S = None
        self.I = None

    def _calc(self, ref_point, F):
        if len(F) == 0:
            self.I, self.S = [], []
            return 0.0, np.zeros(0)

        F = np.minimum(self.ref_point, self.F)
        I = np.lexsort((-F[:, 1], F[:, 0]))
        S = np.argsort(I)

        hv, hvc = super()._calc(ref_point, F[I])
        hvc = hvc[S]
        self.I, self.S = I, S

        return hv, hvc

    # this very efficiently just recomputes hvc of the points necessary
    # however has not shown to be much faster because of reindexing
    # def delete(self, k):
    #     assert k < len(self.F)
    #
    #     F, I, S, hv, hvc = self.F, self.I, self.S, self.hv, self.hvc
    #
    #     hv -= hvc[k]
    #
    #     i = S[k]
    #
    #     S = np.delete(S, k, axis=0)
    #     S[S > i] -= 1
    #
    #     I = np.delete(I, i, axis=0)
    #     I[I > k] -= 1
    #
    #     F = np.delete(F, k, axis=0)
    #     hvc = np.delete(self.hvc, k, axis=0)
    #
    #     v = [I[i] if 0 <= i < len(I) else None for i in np.arange(i-2, i+2)]
    #     left, middle, right = v[0], v[1:-1], v[-1]
    #
    #     middle = [e for e in middle if e is not None]
    #
    #     if len(middle) > 0:
    #
    #         hvc[middle] = hvc_2d_fast(self.ref_point,
    #                                   F[middle],
    #                                   left=F[left] if left is not None else None,
    #                                   right=F[right] if right is not None else None,
    #                                   )
    #
    #     self.F, self.I, self.S, self.hv, self.hvc = F, I, S, hv, hvc
