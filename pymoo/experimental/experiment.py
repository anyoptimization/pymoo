import time

import matplotlib.pyplot as plt
import numpy as np

from pymoo.algorithms.nsga3 import NSGA3
from pymoo.experimental.ansga3 import AdaptiveNSGA3
from pymoo.model.termination import MaximumGenerationTermination
from pymoo.util.plotting import plot, animate
from pymoo.util.reference_directions import get_ref_dirs_from_section
from pymop.problems.dtlz import DTLZ4, DTLZ2


def run():
    start_time = time.time()

    problem = DTLZ2(n_var=20, n_obj=3)

    algorithm = AdaptiveNSGA3(ref_dirs=get_ref_dirs_from_section(3, 12))

    res = algorithm.solve(problem,
                          termination=MaximumGenerationTermination(200),
                          seed=2,
                          save_history=True,
                          disp=True)

    X, F, history = res['X'], res['F'], res['history']

    print("--- %s seconds ---" % (time.time() - start_time))

    scatter_plot = True
    save_animation = False

    if scatter_plot:
        plot(F, problem)

    if save_animation:
        animate('%s.mp4' % problem.name(), history, problem)

    ideal_point = np.zeros(len(history))
    nadir_point = np.zeros(len(history))

    for i, entry in enumerate(history):
        ideal_point[i] = np.linalg.norm(entry.survival.ideal_point)
        nadir_point[i] = np.linalg.norm(entry.survival.intercepts + entry.survival.ideal_point)

    #plt.plot(np.arange(len(history)), ideal_point)
    #plt.plot(np.arange(len(history)), nadir_point)
    #plt.show()

    print("test")


if __name__ == '__main__':
    run()
