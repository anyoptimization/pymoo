from pymoo.algorithms.moo.kgb import KGB
from pymoo.core.callback import CallbackCollection
from pymoo.optimize import minimize
from pymoo.problems.dyn import TimeSimulation
from pymoo.problems.dynamic.df import DF1

from pymoo.visualization.video.callback_video import ObjectiveSpaceAnimation

problem = DF1(taut=2, n_var=2)

algorithm = KGB()

simulation = TimeSimulation()

res = minimize(problem,
               algorithm,
               termination=('n_gen', 100),
               callback=CallbackCollection(ObjectiveSpaceAnimation(), simulation),
               seed=1,
               verbose=True)
