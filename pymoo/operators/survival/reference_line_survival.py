import numpy as np

from pymoo.model import random
from pymoo.model.survival import Survival
from pymoo.util.misc import normalize
from pymoo.util.non_dominated_rank import NonDominatedRank


class ReferenceLineSurvival(Survival):
    def __init__(self, ref_lines):
        super().__init__()
        self.ref_lines = ref_lines

    def _do(self, pop, size):

        fronts = NonDominatedRank.calc_as_fronts(pop.F, pop.G)

        # all indices to survive
        survival = []

        for front in fronts:
            if len(survival) + len(front) > size:
                break
            survival.extend(front)

        # filter the front to only relevant entries
        pop.filter(survival + front)
        survival = list(range(0, len(survival)))
        last_front = list(range(len(survival), pop.size()))

        # if the last front needs to be splitted
        n_remaining = size - len(survival)
        if n_remaining > 0:

            # TODO: Add the Das Dennis stuff here for normalization
            ideal = np.min(pop.F, axis=0)
            nadir = np.max(pop.F, axis=0)

            N = normalize(pop.F, x_min=ideal, x_max=nadir)

            dist_matrix = calc_perpendicular_dist_matrix(N, self.ref_lines)
            niche_of_individuals = np.argmin(dist_matrix, axis=1)
            min_dist_matrix = dist_matrix[np.arange(len(dist_matrix)),niche_of_individuals]

            # for each reference direction the niche count
            niche_count = np.zeros(len(self.ref_lines))
            for i in niche_of_individuals[survival]:
                niche_count[i] += 1

            while n_remaining > 0:

                # all niches where new individuals can be assigned to
                next_niches_list = np.unique(niche_of_individuals[last_front])

                # pick a niche with minimum assigned individuals - break tie if necessary
                next_niche_count = niche_count[next_niches_list]
                next_niche = np.where(next_niche_count == next_niche_count.min())[0]
                next_niche = next_niche[random.randint(0, len(next_niche))]
                next_niche = next_niches_list[next_niche]

                # indices of individuals in last front to assign niche to
                next_ind = np.array(last_front)[np.where(niche_of_individuals[last_front] == next_niche)[0]]

                if len(next_ind) == 1:
                    next_ind = next_ind[0]
                elif niche_count[next_niche] == 0:
                    next_ind = next_ind[np.argmin(min_dist_matrix[next_ind])]
                else:
                    next_ind = next_ind[random.randint(0, len(next_ind))]

                survival.append(next_ind)
                last_front.remove(next_ind)
                niche_count[next_niche] += 1
                n_remaining -= 1

        # now truncate the population
        pop.filter(survival)

        return pop


def calc_perpendicular_dist_matrix(N, ref_lines):
    n = np.tile(ref_lines, (len(N), 1))
    p = np.repeat(N, len(ref_lines), axis=0)
    a = np.zeros((len(p), N.shape[1]))

    val = (a-p) - ((a-p)*n)*n
    dist = np.linalg.norm(val, axis=1)
    matrix = np.reshape(dist, (len(N), len(ref_lines)))

    return matrix

