import numpy as np


def calc_as_fronts_pygmo(pop):

    import pygmo as pg

    if pop.size() == 0:
        return np.array()

    # if there are constraints
    if pop.G.shape[1] > 0:
        constr = np.sum(pop.G[pop.G > 0], axis=1)

        uconstr = np.unique(constr)
        np.sort(uconstr)

        res = []

        for violation in uconstr:
            indices = np.where(constr == violation)[0]

            if len(indices) > 1:
                to_sort = pop.F[indices]
                fronts = pg.fast_non_dominated_sorting(to_sort)[0]
                for front in fronts:
                    mapped = [indices[front[j]] for j in range(len(front))]
                    res.append(np.array(mapped))
            else:
                res.append(indices)

        return res

    else:
        return pg.fast_non_dominated_sorting(pop.F)[0]