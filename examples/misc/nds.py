from timeit import timeit

import numpy as np

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

F = np.random.random((10000, 5))

nds = NonDominatedSorting(method="fast_non_dominated_sort").do(F, only_non_dominated_front=True)
print("Fast Non-Dominated Sorting")
print("result", np.sort(nds))
print("time", timeit("NonDominatedSorting(method='fast_non_dominated_sort').do(F, only_non_dominated_front=True)", number=10, globals=globals()))

nds = NonDominatedSorting(method='efficient_non_dominated_sort').do(F, only_non_dominated_front=True)
print("Fast Non-Dominated Sorting")
print("result", np.sort(nds))
print("time", timeit("NonDominatedSorting(method='efficient_non_dominated_sort').do(F, only_non_dominated_front=True)", number=10, globals=globals()))

