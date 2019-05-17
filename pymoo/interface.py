"""
This class provide an interface for other libraries to specific modules. For example, the evolutionary operations
can be used easily just by calling a function and providing the lower and upper bounds of the problem.

"""
import numpy as np

from pymoo.model.population import Population
from pymop import Problem


def get_problem_func(n_var, xl, xu, type_var):
    class P(Problem):
        def __init__(self) -> None:
            super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=xl, xu=xu, type_var=type_var)

    return P


def sample(sampling, n_samples, n_var, xl=0, xu=1, type_var=np.double, **kwargs):
    problem = get_problem_func(n_var, xl, xu, type_var)(**kwargs)
    return sampling.sample(problem, Population(), n_samples, **kwargs).get("X")


def crossover(crossover, a, b, c=None, xl=0, xu=1, type_var=np.double, **kwargs):
    n = a.shape[0]
    _pop = Population().new("X", a).merge(Population().new("X", b))
    _P = np.column_stack([np.arange(n), np.arange(n) + n])

    if c is not None:
        _pop = _pop.merge(Population().new("X", c))
        _P = np.column_stack([_P, np.arange(n) + 2 * n])

    problem = get_problem_func(a.shape[1], xl, xu, type_var)(**kwargs)
    return crossover.do(problem, _pop, _P, **kwargs).get("X")


def mutation(mutation, X, xl=0, xu=1, type_var=np.double, **kwargs):
    problem = get_problem_func(X.shape[1], xl, xu, type_var)(**kwargs)
    return mutation.do(problem, Population().new("X", X), **kwargs).get("X")


if __name__ == "__main__":
    from pymoo.interface import crossover
    from pymoo.factory import get_crossover
    import numpy as np
    import matplotlib.pyplot as plt


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
    show((off[n_matings:] != b[0]))
    print(off[n_matings:])

    print("K Point Crossover (k=4)")
    off = crossover(get_crossover("bin_k_point", n_points=4), a, b)
    show((off[:n_matings] != a[0]))

