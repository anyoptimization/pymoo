import pygmo
import numpy as np


def denormalize(x, min, max):
    return x * (max - min) + min


def normalize(x, x_min=None, x_max=None, return_bounds=False):
    if x_min is None:
        x_min = np.min(x, axis=0)
    if x_max is None:
        x_max = np.max(x, axis=0)

    res = (x - x_min) / (x_max - x_min)
    if not return_bounds:
        return res
    else:
        return res, x_min, x_max


def print_pop(pop, rank, crowding, sorted_idx, n):
    for i in range(n):
        print(i, pop[sorted_idx[i]].f, pop[sorted_idx[i]].c, rank[sorted_idx[i]], crowding[sorted_idx[i]])
    print('---------')



def get_front_by_index(f):
    return pygmo.fast_non_dominated_sorting(f)[0][0]


def get_front(f):
    return f[get_front_by_index(f)]

