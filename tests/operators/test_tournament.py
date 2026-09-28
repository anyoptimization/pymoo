import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import comp_by_cv_and_fitness
from pymoo.core.population import Population


def population(F, CV):
    F = np.array(F, dtype=float)[:, None]
    pop = Population.new(X=F.copy(), F=F)
    pop.set("CV", np.array(CV, dtype=float)[:, None])
    return pop


def winners(pop, P):
    return comp_by_cv_and_fitness(pop, np.array(P), random_state=np.random.default_rng(1)).ravel().tolist()


def test_comp_by_cv_and_fitness_all_competitors():
    pop = population(F=[3.0, 2.0, 1.0, 4.0], CV=[0.0, 0.0, 0.0, 0.0])

    assert winners(pop, [[0, 1], [2, 3]]) == [1, 2]
    assert winners(pop, [[0, 1, 2], [2, 1, 0], [3, 0, 1], [0, 1, 3]]) == [2, 2, 1, 1]
    assert winners(pop, [[0, 1, 2, 3], [3, 0, 2, 1]]) == [2, 2]


def test_comp_by_cv_and_fitness_all_competitors_constrained():
    # feasible beats infeasible, then the smaller constraint violation wins
    pop = population(F=[1.0, 3.0, 2.0, 0.0], CV=[0.5, 0.0, 0.0, 0.1])

    assert winners(pop, [[0, 3], [1, 0]]) == [3, 1]
    assert winners(pop, [[0, 1, 2], [0, 3, 1], [3, 0, 0]]) == [2, 1, 3]
