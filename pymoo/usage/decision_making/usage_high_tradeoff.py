# START high_tradeoff_2d
import os

import numpy as np

from pymoo.configuration import get_pymoo
from pymoo.factory import get_decision_making
from pymoo.visualization.scatter import Scatter

pf = np.loadtxt(os.path.join(get_pymoo(), "pymoo", "usage", "decision_making", "knee-2d.out"))
dm = get_decision_making("high-tradeoff")

I = dm.do(pf)

plot = Scatter()
plot.add(pf, alpha=0.2)
plot.add(pf[I], color="red", s=100)
plot.show()

# END high_tradeoff_2d



# START high_tradeoff_3d

pf = np.loadtxt(os.path.join(get_pymoo(), "pymoo", "usage", "decision_making", "knee-3d.out"))
dm = get_decision_making("high-tradeoff")

I = dm.do(pf)

plot = Scatter(angle=(10, 140))
plot.add(pf, alpha=0.2)
plot.add(pf[I], color="red", s=100)
plot.show()
# END high_tradeoff_3d

