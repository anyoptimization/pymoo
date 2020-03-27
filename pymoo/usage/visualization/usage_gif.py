import matplotlib.pyplot as plt

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT1
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.visualization.pcp import PCP
from pymoo.visualization.scatter import Scatter
from pyrecorder.recorders.gif import GIF
from pyrecorder.video import Video

problem = ZDT1(n_var=6)

algorithm = NSGA2()

ret = minimize(problem,
               algorithm,
               termination=('n_gen', 61),
               seed=1,
               save_history=True,
               verbose=False)

print(ret.F)

with Video(GIF("animation.gif")) as vid:
    for algorithm in ret.history:

        if algorithm.n_gen % 5 == 0:
            X, F = algorithm.pop.get("X", "F")
            nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
            other = [k for k in range(len(F)) if k not in nds]

            fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
            fig.suptitle("%s - %s - Gen %s" % ("ZDT1", "NSGA2", algorithm.n_gen), fontsize=16)

            pcp = PCP(ax=ax1, bounds=(problem.xl, problem.xu))
            pcp.set_axis_style(color="black", alpha=0.7)
            pcp.add(X[other], color="blue", linewidth=0.5)
            pcp.add(X[nds], color="red", linewidth=2)
            pcp.do()

            sc = Scatter(ax=ax2)
            sc.add(F[other], color="blue")
            sc.add(F[nds], color="red")
            sc.add(problem.pareto_front(), plot_type="line", color="black")
            sc.do()

            vid.record()
