# START load_data
import numpy as np

np.random.seed(1234)
F = np.random.random((4, 6))
# END load_data

# START heatmap
from pymoo.analytics.visualization.heatmap import heatmap
heatmap().add(F).show()
# END heatmap

# START heatmap_custom
plot = heatmap(title=("Optimization", {'pad': 15}),
               cmap="Oranges_r",
               y_labels=["Solution A", "Solution B", "Solution C", "Solution D"],
               labels=["profit", "cost", "sustainability", "environment", "satisfaction", "time"])
plot.add(F)
plot.show()

# END heatmap_custom

# START heatmap_custom_more
F = np.random.random((30, 6))

plot = heatmap(figsize=(10,30),
               bound=[0,1],
               order_by_objectives=0,
               y_labels=None,
               labels=["profit", "cost", "sustainability", "environment", "satisfaction", "time"],
               cmap="Greens_r")

plot.add(F, aspect=0.2)
plot.show()
# END heatmap_custom_more
