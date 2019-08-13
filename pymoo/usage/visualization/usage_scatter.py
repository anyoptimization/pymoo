# START scatter2d
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_problem, get_reference_directions

F = get_problem("zdt3").pareto_front()
Scatter().add(F).show()
# END scatter2d


# START scatter2d_custom
F = get_problem("zdt3").pareto_front(use_cache=False, flatten=False)
plot = Scatter()
plot.add(F, s=30, facecolors='none', edgecolors='r')
plot.add(F, plot_type="line", color="black", linewidth=2)
plot.show()
# END scatter2d_custom


# START scatter3d
ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
F = get_problem("dtlz1").pareto_front(ref_dirs)

plot = Scatter()
plot.add(F)
plot.show()
# END scatter3d

# START scatter4d
import numpy as np
F = np.random.random((30, 4))

plot = Scatter(tight_layout=True)
plot.add(F, s=10)
plot.add(F[10], s=30, color="red")
plot.show()
# END scatter4d
