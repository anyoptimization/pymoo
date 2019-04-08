import matplotlib.pyplot as plt
import numpy as np

from pymoo.experimental.emo.true import ReferenceDirectionSurvivalTrue
from pymoo.optimize import minimize
from pymoo.util import plotting
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop import ZDT1


class ScaledZDT1(ZDT1):

    def _calc_pareto_front(self, n_pareto_points=1000):
        pf = super()._calc_pareto_front(2000) * np.array([1.0, 10.0])
        #cd = calc_crowding_distance(pf)
        #pf = pf[np.argsort(cd)[::-1][:200]]
        return pf - 500

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"] = out["F"] * np.array([1.0, 10.0]) - 500


problem = ScaledZDT1()

ref_dirs = UniformReferenceDirectionFactory(2, n_points=100).do()

pf = problem.pareto_front()

res = minimize(problem,
               method='nsga3',
               method_args={
                   'pop_size': 100,
                   'ref_dirs': ref_dirs,
                   #'survival': ReferenceDirectionSurvivalTrue(ref_dirs, np.array([[1.0, 0], [0, 1.0]]))
               },
               termination=('n_gen', 400),
               pf=pf,
               seed=1,
               disp=True)

plotting.plot(pf, res.F, labels=["Pareto-front", "NSGA-III - No Normalization"], show=False)
plt.legend()
plt.savefig("no_normalization.pdf")