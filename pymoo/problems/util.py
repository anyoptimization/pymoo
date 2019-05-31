import os

import autograd.numpy as anp


def load_pareto_front_from_file(fname):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(current_dir, "pf", "%s" % fname)
    if os.path.isfile(fname):
        pf = anp.loadtxt(fname)
        return pf[pf[:, 0].argsort()]


def binomial(n, k):
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

