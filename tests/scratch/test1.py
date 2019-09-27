import os

import numpy as np

from pymoo.configuration import get_pymoo
from pymoo.decision_making.high_tradeoff import HighTradeoffPoints
from pymoo.decision_making.high_tradeoff_inverted import HighTradeoffPointsInverted
from pymoo.visualization.scatter import Scatter

pf = np.loadtxt(os.path.join(get_pymoo(), "pymoo", "usage", "decision_making", "knee-2d.out"))
dm = HighTradeoffPoints()

I = dm.do(pf)

plot = Scatter(angle=(0, 0))
plot.add(pf, alpha=0.2)
plot.add(pf[I], color="red", s=100)
plot.show()
