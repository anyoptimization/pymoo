# START load_data

from pymoo.factory import get_problem, get_reference_directions

ref_dirs = get_reference_directions("uniform", 6, n_partitions=5)
F = get_problem("dtlz1").pareto_front(ref_dirs)
# END load_data

# START star
from pymoo.visualization.star_coordinate import StarCoordinate

StarCoordinate().add(F).show()
# END star

# START star_custom
plot = StarCoordinate(title="Optimization",
                      legend=(True, {'loc': "upper left", 'bbox_to_anchor': (-0.1, 1.08, 0, 0)}),
                      labels=["profit", "cost", "sustainability", "environment", "satisfaction", "time"],
                      axis_style={"color": "blue", 'alpha': 0.7},
                      arrow_style={"head_length": 0.015, "head_width": 0.03})
plot.add(F, color="grey", s=20)
plot.add(F[65], color="red", s=70, label="Solution A")
plot.add(F[72], color="green", s=70, label="Solution B")
plot.show()
# END star_custom
