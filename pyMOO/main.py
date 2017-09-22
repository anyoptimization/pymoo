import os

import matplotlib.pyplot as plt

from pyMOO.algorithms.nsga.nsga import NSGA
from pyMOO.problems.dtlz import DTLZ1
from pyMOO.problems.zdt import ZDT1


def write_final_pop_obj(pop, run):
    f_name = os.path.join('results', problem.__class__.__name__ + '_RUN' + str(run) + str('.out'))
    f = open(f_name, 'w')
    for ind in pop:
        f.write('%f \t %f\n' % (ind.f[0], ind.f[1]))
    f.close()


if __name__ == '__main__':

    problem = DTLZ1()
    pop = NSGA(pop_size=100).solve(problem, n_eval=30000)

    x = [pop[i].f[0] for i in range(len(pop))]
    y = [pop[i].f[1] for i in range(len(pop))]
    plt.scatter(x, y)
    plt.show()

    # write_final_pop_obj(pop,1)
