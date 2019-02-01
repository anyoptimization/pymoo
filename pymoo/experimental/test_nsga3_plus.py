import numpy as np

from pymoo.experimental.nsga3_plus import NSGA3Plus
from pymoo.model.termination import MaximumGenerationTermination
from pymoo.util import plotting
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.factory import get_problem

problem = get_problem("c1dtlz3", n_var=12, n_obj=2)

# create the reference directions to be used for the optimization
#ref_dirs = UniformReferenceDirectionFactory(3, n_points=91).do()
ref_dirs = UniformReferenceDirectionFactory(2, n_points=100).do()

# create the pareto front for the given reference lines
pf = problem.pareto_front(ref_dirs)

algorithm = NSGA3Plus(pop_size=100, ref_dirs=ref_dirs)

res = algorithm.solve(problem,
                      MaximumGenerationTermination(1000),
                      pf=pf,
                      save_history=True,
                      disp=True)



from pymoo.util.plotting import animate as func_animtate


def callback(ax, *args):
    pass

    # 1
    #ax.set_xlim(0, 2)
    #ax.set_ylim(-1.2, 0.1)

    #2
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0.2, 1.2)


plotting.plot(res.F)

H = np.concatenate([e.pop.get("F")[None, ...] for e in res.history], axis=0)
func_animtate('nsga3_plus.mp4' % problem.name(), H, problem, func_iter=callback)
