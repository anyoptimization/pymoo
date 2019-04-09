import matplotlib.pyplot as plt
import numpy as np

from pymoo.experimental.emo.true import ReferenceDirectionSurvivalTrue
from pymoo.optimize import minimize
from pymoo.util import plotting
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.factory import get_problem, ScaledProblem, DTLZ1
from pymop.problem import Problem


test = "sasfsfsf"

a = 5
b = 10

print("sfsdf")

for i in range(5):
    print(i)


class InvertedDTLZ1(DTLZ1):

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)

        super()._evaluate(x, out, *args, **kwargs)
        out["F"] = 0.5 * (1 + g[:, None]) - out["F"]


#problem = get_problem("dtlz1", None, 3, k=5)
problem = InvertedDTLZ1(n_obj=3)

ref_dirs = UniformReferenceDirectionFactory(3, n_points=91).do()

pf = problem.pareto_front(ref_dirs)

res = minimize(problem,
               method='nsga3',
               method_args={
                   'pop_size': 100,
                   'ref_dirs': ref_dirs,
                   },
               termination=('n_gen', 500),
               #pf=pf,
               seed=1,
               disp=True)

closest_to_ref_dir = res.opt.get("closest")
plotting.plot(res.F[closest_to_ref_dir,:], labels=["NSGA-III"], show=False)
#plotting.plot(pf, res.F, labels=["Pareto-front", "NSGA-III"], show=False)
plt.legend()
plt.show()