"""Bitflip mutation operator for binary variables."""

import numpy as np

from pymoo.core.mutation import Mutation


class BitflipMutation(Mutation):
    """Bitflip mutation operator for binary chromosomes.

    Flips bits in binary genotypes with an independent probability per locus.

    In evolutionary computation literature, mutation probability (:math:`p_m`)
    almost universally refers to the per-variable (per-bit) flip rate (typically
    :math:`1 / n_{\\text{var}}`). In PyMoo, mutation operates on two hierarchical levels:

    1. **Chromosome level (``prob``)**: The probability that an offspring individual
       undergoes the mutation operator at all. Defaults to ``1.0`` (all offspring are evaluated).
    2. **Gene level (``prob_var``)**: The probability that each individual bit is flipped.
       Defaults to ``min(0.5, 1 / n_var)``.

    .. warning::
        **Compound Probability Trap:**
        Do not pass your per-locus mutation rate (e.g. :math:`1/n`) as the first positional
        argument ``prob`` without specifying ``prob_var``:

        .. code-block:: python

            # WRONG: Effective per-bit rate degrades to (1/n) * (1/n) = 1/n^2!
            mutation = BitflipMutation(1.0 / n_var)

            # CORRECT: Evaluates every offspring with an independent 1/n per-bit rate:
            mutation = BitflipMutation(prob=1.0, prob_var=1.0 / n_var)

            # ALSO CORRECT (relies on PyMoo's default prob_var = 1/n_var):
            mutation = BitflipMutation(prob=1.0)

    Parameters
    ----------
    prob : float, optional (default=1.0)
        Probability that an individual undergoes the bitflip mutation routine.
        In canonical genetic algorithms, this is set to ``1.0`` so that every offspring
        is subject to per-locus flipping.
    prob_var : float, optional (default=None)
        Probability of flipping each individual bit. If ``None``, defaults to
        ``min(0.5, 1.0 / problem.n_var)``, which satisfies the theoretical expectation of
        :math:`\\mathbb{E}[\\text{flips}] = 1.0` bit per individual.

    Examples
    --------
    >>> from pymoo.operators.mutation.bitflip import BitflipMutation
    >>> # Standard GA configuration (E[flips] = 1.0 bit per chromosome):
    >>> mutation = BitflipMutation(prob=1.0)
    >>>
    >>> # Explicit per-locus rate (e.g. 5% flip probability per bit):
    >>> mutation = BitflipMutation(prob=1.0, prob_var=0.05)
    """

    def __init__(self, prob=1.0, prob_var=None, **kwargs):
        super().__init__(prob=prob, prob_var=prob_var, **kwargs)

    def _do(self, problem, X, random_state=None, **kwargs):
        prob_var = self.get_prob_var(problem, size=(len(X), 1))
        Xp = np.copy(X)
        flip = random_state.random(X.shape) < prob_var
        Xp[flip] = ~X[flip]
        return Xp


class BFM(BitflipMutation):
    pass
