# START load_data

from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

ref_dirs = get_reference_directions("uniform", 6, n_partitions=5)
F = get_problem("dtlz1").pareto_front(ref_dirs)
# END load_data

# START radviz
from pymoo.visualization.radviz import Radviz
Radviz().add(F).show()
# END radviz

# START radviz_custom
plot = Radviz(title="Optimization",
              legend=(True, {'loc': "upper left", 'bbox_to_anchor': (-0.1, 1.08, 0, 0)}),
              labels=["profit", "cost", "sustainability", "environment", "satisfaction", "time"],
              endpoint_style={"s": 70, "color": "green"})
plot.set_axis_style(color="black", alpha=1.0)
plot.add(F, color="grey", s=20)
plot.add(F[65], color="red", s=70, label="Solution A")
plot.add(F[72], color="blue", s=70, label="Solution B")
plot.show()
# END radviz_custom
