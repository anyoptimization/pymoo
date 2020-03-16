# START example
import numpy as np
from pyrecorder.recorders.file import File
from pyrecorder.video import Video

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter

vid = Video(File("example.mp4"))

for k in range(10):
    X = np.random.random((100, 2))
    Scatter(title=str(k)).add(X).do()
    vid.record()

vid.close()
# END example


# START ga
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("zdt1")

algorithm = NSGA2(pop_size=100, eliminate_duplicates=True)

ret = minimize(problem,
               algorithm,
               termination=('n_gen', 100),
               seed=1,
               save_history=True,
               verbose=False)

# use the video writer as a resource
with Video(File("ga.mp4")) as vid:

    # for each algorithm object in the history
    for entry in ret.history:
        sc = Scatter(title=("Gen %s" % entry.n_gen))
        sc.add(entry.pop.get("F"))
        sc.add(entry.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
        sc.do()

        # finally record the current visualization to the video
        vid.record()
# END ga



