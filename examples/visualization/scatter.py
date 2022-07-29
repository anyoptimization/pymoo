import numpy as np

from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

np.random.seed(1)

F = get_problem("zdt3").pareto_front()
Scatter().add(F[:, [0]]).show()

F = get_problem("zdt3").pareto_front()
Scatter().add(F).show()

F = get_problem("zdt3").pareto_front(use_cache=False, flatten=False)
plot = Scatter()
plot.add(F, s=30, facecolors='none', edgecolors='r')
plot.add(F, plot_type="line", color="black", linewidth=2)
plot.show()

ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
F = get_problem("dtlz1").pareto_front(ref_dirs)

plot = Scatter(plot_3d=True)
plot.add(F)
plot.show()


plot = Scatter(plot_3d=False)
plot.add(F)
plot.show()

F = np.random.random((30, 4))

plot = Scatter(tight_layout=True)
plot.add(F, s=10)
plot.add(F[10], s=30, color="red")
plot.show()
