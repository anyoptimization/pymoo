# START load_data
import numpy as np

np.random.seed(3)

ideal_point = np.array([0.15, 0.1, 0.2, 0.1, 0.1])
nadir_point = np.array([0.85, 0.9, 0.95, 0.9, 0.85])

F = np.random.random((1, 5)) * (nadir_point - ideal_point) + ideal_point
print(F)
# END load_data


# START radar
from pymoo.visualization.radar import Radar

plot = Radar(bounds=[ideal_point, nadir_point], normalize_each_objective=False)
plot.add(F)
plot.show()
# END radar


# START radar_norm
plot = Radar(bounds=[ideal_point, nadir_point])
plot.add(F)
plot.show()
# END radar_norm

# START radar_custom
F = np.random.random((6, 5)) * (nadir_point - ideal_point) + ideal_point

plot = Radar(bounds=[ideal_point, nadir_point],
             axis_style={"color": 'blue'},
             point_style={"color": 'red', 's': 30})
plot.add(F[:3], color="red", alpha=0.8)
plot.add(F[3:], color="green", alpha=0.8)
plot.show()
# END radar_custom
