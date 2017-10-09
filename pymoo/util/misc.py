import numpy as np


def get_x(pop):
    return np.array([pop[i].x for i in range(len(pop))])


def get_f(pop):
    return np.array([pop[i].f for i in range(len(pop))])


def evaluate(evaluator, problem, pop):
    for ind in pop:
        ind.f, ind.g = evaluator.eval(problem, ind.x)


def print_pop(pop, rank, crowding, sorted_idx, n):
    for i in range(n):
        print(i, pop[sorted_idx[i]].f, rank[sorted_idx[i]], crowding[sorted_idx[i]])
    print('---------')

