from timeit import timeit
# noinspection PyUnresolvedReferences
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import numpy as np

# generate random data samples
F = np.random.random((1000, 2))

# use fast non-dominated sorting
res = timeit("NonDominatedSorting(method=\"fast_non_dominated_sort\").do(F)", number=10, globals=globals())
print(f"Fast ND sort takes {res} seconds")

# # use efficient non-dominated sorting with sequential search, this is the default method
# res = timeit("NonDominatedSorting(method=\"efficient_non_dominated_sort\").do(F, strategy=\"sequential\")", number=10,
#              globals=globals())
# print(f"Efficient ND sort with sequential search (ENS-SS) takes {res} seconds")
#
#
# res = timeit("NonDominatedSorting(method=\"efficient_non_dominated_sort\").do(F, strategy=\"binary\")", number=10,
#              globals=globals())
# print(f"Efficient ND sort with binary search (ENS-BS) takes {res} seconds")
