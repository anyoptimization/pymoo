# START load_data

from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

ref_dirs = get_reference_directions("das-dennis", 6, n_partitions=5) * [2, 4, 8, 16, 32, 64]
F = get_problem("dtlz1").pareto_front(ref_dirs)
# END load_data

# START pcp
from pymoo.visualization.pcp import PCP
PCP().add(F).show()
# END pcp


# START pcp_highlight
plot = PCP()
plot.set_axis_style(color="grey", alpha=0.5)
plot.add(F, color="grey", alpha=0.3)
plot.add(F[50], linewidth=5, color="red")
plot.add(F[75], linewidth=5, color="blue")
plot.show()
# END pcp_highlight


# START pcp_other
plot = PCP(title=("Run", {'pad': 30}),
           n_ticks=10,
           legend=(True, {'loc': "upper left"}),
           labels=["profit", "cost", "sustainability", "environment", "satisfaction", "time"]
           )

plot.set_axis_style(color="grey", alpha=1)
plot.add(F, color="grey", alpha=0.3)
plot.add(F[50], linewidth=5, color="red", label="Solution A")
plot.add(F[75], linewidth=5, color="blue", label="Solution B")
plot.show()
# END pcp_other

# START pcp_norm
plot.reset()
plot.normalize_each_axis = False
plot.bounds = [[1, 1, 1, 2, 2, 5], [32, 32, 32, 32, 32, 32]]
plot.show()
# END pcp_norm
