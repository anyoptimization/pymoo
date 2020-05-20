import numpy as np

from pymoo.model.crossover import Crossover


def random_sequence(n):
    start, end = np.sort(np.random.choice(n, 2, replace=False))
    return tuple([start, end])


# Implementation based on http://www.dmi.unict.it/mpavone/nc-cs/materiale/moscato89.pdf
def ox(receiver, donor, seq=None, shift=True):
    assert len(donor) == len(receiver)

    seq = seq if not None else random_sequence(len(receiver))
    start, end = seq

    donation = np.copy(donor[start:end + 1])
    donation_as_set = set(donation)

    # the final value to be returned
    y = []

    for k in range(len(receiver)):

        # do the shift starting from the swapped sequence - as proposed in the paper
        i = k if not shift else (start + k) % len(receiver)
        v = receiver[i]

        if v not in donation_as_set:
            y.append(v)

    # now insert the donation at the right place
    y = np.concatenate([y[:start], donation, y[start:]]).astype(copy=False, dtype=np.int)

    return y


class OrderCrossover(Crossover):

    def __init__(self, **kwargs):
        super().__init__(2, 2, **kwargs)

    def _do(self, problem, X, **kwargs):
        n_matings = X.shape[1]
        Y = np.full(X.shape, 0, dtype=problem.type_var)

        for i in range(n_matings):
            a, b = X[:, i, :]
            n = len(a)

            # define the sequence to be used for crossover
            start, end = random_sequence(n)

            # if the receiver should be shifted and actually start with the original sequence
            shift = np.random.rand() < 0.5

            Y[0, i, :] = ox(a, b, seq=(start, end), shift=shift)
            Y[1, i, :] = ox(b, a, seq=(start, end), shift=shift)

        return Y
