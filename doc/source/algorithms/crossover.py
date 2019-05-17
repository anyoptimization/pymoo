##START SBX1

from pymoo.interface import crossover
from pymoo.factory import get_crossover
import numpy as np
import matplotlib.pyplot as plt


def show(eta_cross):
    a, b = np.full((5000, 1), 0.2), np.full((5000, 1), 0.8)
    off = crossover(get_crossover("real_sbx", prob=1.0, eta=eta_cross, prob_per_variable=1.0), a, b)

    plt.hist(off, range=(0, 1), bins=200, density=True, color="red")
    plt.show()


show(5)


##END SBX1


##START POINT

def example_parents(n_matings, n_var):
    a = np.arange(n_var)[None, :].repeat(n_matings, axis=0)
    b = a + n_var
    return a, b


def show(M):
    plt.figure(figsize=(4, 4))
    plt.imshow(M, cmap='Greys', interpolation='nearest')
    plt.show()


n_matings, n_var = 100, 100
a, b = example_parents(n_matings, n_var)

print("One Point Crossover")
off = crossover(get_crossover("bin_one_point"), a, b)
show((off[:n_matings] != a[0]))

print("Two Point Crossover")
off = crossover(get_crossover("bin_two_point"), a, b)
show((off[:n_matings] != a[0]))

print("K Point Crossover (k=4)")
off = crossover(get_crossover("bin_k_point", n_points=4), a, b)
show((off[:n_matings] != a[0]))

##END POINT
