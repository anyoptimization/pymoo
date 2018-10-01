import time

import numpy as np

from pymoo.algorithms.nsga3 import NSGA3
from pymoo.experimental.normalization.c_max_extremes import MaxExtremesReferenceLineSurvival
from pymoo.experimental.normalization.c_max_non_dom import MaxNonDominatedReferenceLineSurvival
from pymoo.experimental.normalization.perfect import PerfectReferenceLineSurvival
from pymoo.experimental.normalization.propose import ProposeReferenceLineSurvival
from pymoo.model.termination import MaximumGenerationTermination
from pymoo.util.plotting import plot, animate
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.problems.dtlz import DTLZ1


def run():
    start_time = time.time()

    #problem = ScaledProblem(DTLZ2(n_var=12, n_obj=3), 10)
    problem = DTLZ1(n_var=8, n_obj=3)
    problem.n_pareto_points = 91

    #plot_3d(problem.pareto_front())
    #plt.xlabel("X")
    #plt.ylabel("Y")
    #plt.zlabel("Z")
    #plt.show()

    ref_dirs = UniformReferenceDirectionFactory(n_dim=3, n_points=92).do()

    algorithm = NSGA3(ref_dirs)
    algorithm.survival = PerfectReferenceLineSurvival(ref_dirs, problem.ideal_point(), problem.nadir_point())

    res = algorithm.solve(problem,
                          termination=MaximumGenerationTermination(1000),
                          seed=69,
                          save_history=True,
                          disp=True)

    X, F, history = res['X'], res['F'], res['history']

    np.savetxt('yash_dtl1_5obj', F)

    print("--- %s seconds ---" % (time.time() - start_time))

    F = (F - problem.ideal_point()) / (problem.nadir_point() - problem.ideal_point())

    scatter_plot = True
    save_animation = False

    if scatter_plot:
        plot(F, problem)

    if save_animation:
        animate('%s.mp4' % problem.name(), history, problem)


if __name__ == '__main__':
    run()
