# START example
import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.video import Video

vid = Video("example.mp4")

for k in range(10):
    X = np.random.random((100, 2))
    Scatter(title=str(k)).add(X).do()
    vid.snap(duration=1)

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


def my_plot(algorithm):
    Scatter(title=("Gen %s" % algorithm.n_gen)).add(algorithm.pop.get("F")).do()

Video.from_iteratable("ga.mp4", ret.history, my_plot)
# END ga



