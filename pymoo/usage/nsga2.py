

import numpy as np

from pymoo.optimize import minimize
from pymoo.util import plotting
from pymop.factory import get_problem

# create the optimization problem
problem = get_problem("zdt1")
pf = problem.pareto_front()

res = minimize(problem,
               method='nsga2',
               method_args={'pop_size': 100},
               termination=('n_gen', 200),
               pf=pf,
               save_history=True,
               disp=True)

plot = True
if plot:
    plotting.plot(pf, res.F, labels=["Pareto-front", "F"])

# set true if you want to save a video
animate = False
if animate:
    from pymoo.util.plotting import animate as func_animtate
    H = np.concatenate([e.pop.get("F")[None, :] for e in res.history], axis=0)
    func_animtate('%s.mp4' % problem.name(), H, problem)
