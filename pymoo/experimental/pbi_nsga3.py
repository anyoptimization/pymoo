from pymoo.util import plotting

from pymoo.experimental.pbi import ReferenceDirectionSurvivalPBI
from pymoo.optimize import minimize
from pymoo.util.reference_direction import UniformReferenceDirectionFactory, MultiLayerReferenceDirectionFactory
from pymop.factory import get_problem

import matplotlib.pyplot as plt

import numpy as np

problem = get_problem("dtlz3", n_var=None, n_obj=15, k=10)

n_gen = 2000
pop_size = 136
ref_dirs = MultiLayerReferenceDirectionFactory([
    UniformReferenceDirectionFactory(15, n_partitions=2, scaling=1.0),
    UniformReferenceDirectionFactory(15, n_partitions=1, scaling=0.5)]).do()

# create the pareto front for the given reference lines
pf = problem.pareto_front(ref_dirs)

ideal_point = []
nadir_point = []


def my_callback(algorithm):
    ideal_point.append(np.copy(algorithm.survival.ideal_point))
    nadir_point.append(np.copy(algorithm.survival.nadir_point))


res = minimize(problem,
               method='nsga3',
               method_args={
                   'pop_size': pop_size,
                   'ref_dirs': ref_dirs,
                   'survival': ReferenceDirectionSurvivalPBI(ref_dirs)
               },
               termination=('n_gen', n_gen),
               pf=pf,
               callback=my_callback,
               seed=31,
               disp=True)


#ideal_point = np.vstack(ideal_point)
#error_ideal = np.sum(np.abs(ideal_point - np.min(pf, axis=0)), axis=1)
#plt.figure(0)
#plt.plot(np.arange(n_gen), error_ideal)
#plt.show()


nadir_point = np.vstack(nadir_point)
error_nadir = np.mean(np.abs(nadir_point - np.max(pf, axis=0)), axis=1)
plt.figure(1)
plt.plot(np.arange(n_gen), error_nadir)
plt.show()

