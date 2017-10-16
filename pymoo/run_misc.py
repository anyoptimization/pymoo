import numpy as np

from algorithms.nsao import NSAO
from problems.rastrigin import Rastrigin
from problems.zdt import ZDT3, ZDT1
from util.misc import get_f, perpendicular_dist, uniform_2d_weights
from util.non_dominated_rank import NonDominatedRank
from vendor.pygmo_algorithm import PygmoAlgorithm
from vendor.pygmo_problem import create_pygmo_problem
import pygmo as pg

if __name__ == '__main__':
    problem = ZDT3(n_var=5)

    #a = PygmoAlgorithm(pg.nsga2(gen=100))
    #x, f = a.solve(problem, None, None)


    ref_dir = np.array([1, 1])
    point = np.array([0, 1])
    projection = (np.dot(point, ref_dir) / np.linalg.norm(ref_dir)) * ref_dir
    print(np.linalg.norm(projection - point))
    print(perpendicular_dist(ref_dir, point))

    pop = NSAO(reference_directions=uniform_2d_weights(6)).solve(problem, evaluator=100, seed=1)
    np.savetxt("nsao_ZDT3.out", np.asarray(get_f(pop)), fmt='%.14f')

    front = [pop[i] for i in NonDominatedRank.calc_as_fronts_pygmo(pop)[0]]

    np.savetxt("nsao_ZDT3_front.out", np.asarray(get_f(front)), fmt='%.14f')
    print(get_f(pop))
    print(get_f(front))
