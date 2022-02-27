import numpy as np

from pymoo.core.crossover import Crossover


def random_sequence(n):
    start, end = np.sort(np.random.choice(n, 2, replace=False))
    return tuple([start, end])


def ox(receiver, donor, seq=None, shift=False):
    """
    The Ordered Crossover (OX) as explained in http://www.dmi.unict.it/mpavone/nc-cs/materiale/moscato89.pdf.

    Parameters
    ----------
    receiver : numpy.array
        The receiver of the sequence. The array needs to be repaired after the donation took place.
    donor : numpy.array
        The donor of the sequence.
    seq : tuple (optional)
        Tuple with two problems defining the start and the end of the sequence. Please note in our implementation
        the end of the sequence is included. The sequence is randomly chosen if not provided.

    shift : bool
        Whether during the repair the receiver should be shifted or not. Both version of it can be found in the
        literature.

    Returns
    -------

    y : numpy.array
        The offspring which was created by the ordered crossover.

    """
    assert len(donor) == len(receiver)

    # the sequence which shall be use for the crossover
    seq = seq if not None else random_sequence(len(receiver))
    start, end = seq

    # the donation and a set of it to allow a quick lookup
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
    y = np.concatenate([y[:start], donation, y[start:]]).astype(copy=False, dtype=int)

    return y


class OrderCrossover(Crossover):

    def __init__(self, shift=False, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.shift = shift

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.full((self.n_offsprings, n_matings, n_var), -1, dtype=int)

        for i in range(n_matings):
            a, b = X[:, i, :]
            n = len(a)

            # define the sequence to be used for crossover
            start, end = random_sequence(n)

            Y[0, i, :] = ox(a, b, seq=(start, end), shift=self.shift)
            Y[1, i, :] = ox(b, a, seq=(start, end), shift=self.shift)

        return Y
