import numpy as np

from pymoo.indicators.hv.exact import DynamicHypervolume


def alpha(N, k):
    alpha = np.zeros(N+1)

    for i in range(1, N + 1):
        alpha[i] = np.prod([(k - j) / (N - j) for j in range(1, i)]) / i

    return alpha


def hv_monte_carlo(dom, V, n_dom=None):
    N, n_samples = dom.shape
    if n_dom is None:
        n_dom = dom.sum(axis=0)

    a = alpha(N, N)
    hv = sum([a[n_dom[dom[i]]].sum() for i in range(N)]) / n_samples * V
    return hv


def hvc_monte_carlo(dom, V, n_dom=None, k=1):
    N, n_samples = dom.shape
    if n_dom is None:
        n_dom = dom.sum(axis=0)

    a = alpha(N, k)
    hvc = np.array([(a[n_dom[dom[i]]].sum() / n_samples * V).sum() for i in range(N)])
    return hvc


class ApproximateMonteCarloHypervolume(DynamicHypervolume):

    def __init__(self, ref_point, n_samples=10000, n_exclusive=1, **kwargs) -> None:
        self.n_samples = n_samples
        self.n_exclusive = n_exclusive

        self.V = None
        self.dom = None

        super().__init__(ref_point, **kwargs)

    def _calc(self, ref_point, F):
        (N, M) = F.shape

        ideal = F.min(axis=0)
        V = np.prod(ref_point - ideal)

        S = np.random.uniform(low=ideal, high=ref_point, size=(self.n_samples, M))

        dom = np.array([np.all(F[i] <= S, axis=1) for i in range(N)])

        n_dom = dom.sum(axis=0)
        hv = hv_monte_carlo(dom, V, n_dom=n_dom)
        hvc = hvc_monte_carlo(dom, V, n_dom=n_dom, k=self.n_exclusive)

        self.V = V
        self.dom = dom

        return hv, hvc

    def delete(self, k):
        self.F = np.delete(self.F, k, axis=0)
        self.dom = np.delete(self.dom, k, axis=0)

        self.hv -= self.hvc[k]

        V, dom = self.V, self.dom
        n_dom = dom.sum(axis=0)
        self.hvc = hvc_monte_carlo(dom, V, n_dom=n_dom, k=self.n_exclusive)

