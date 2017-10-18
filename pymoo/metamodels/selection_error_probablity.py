import numpy as np


def calc_sep(F, F_hat):
    n = len(F)
    val = 0.00
    counter = 0
    for i in range(n):
        for j in range(i + 1, n):
            counter += 1
            if (np.all(F[i] > F[j]) and np.all(F_hat[i] < F_hat[j])) or (
                        np.all(F[i] < F[j]) and np.all(F_hat[i] > F_hat[j])):
                val += 1.0
    return val / counter
