import numpy as np

from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.problems import get_problem
from pymoo.indicators.rmetric import RMetric
from pymoo.core.callback import Callback
from pymoo.optimize import minimize

problem = get_problem("zdt1", n_var=30)
pf = problem.pareto_front()
ref_points = np.array([[0.5, 0.2], [0.1, 0.6]])


class MyCallback(Callback):

    def notify(self, algorithm):
        rmetric = RMetric(algorithm.problem, ref_points)
        rigd, rhv = rmetric.do(algorithm.opt.get("F"))
        print(f"R-IGD: {rigd}, R-HV: {rhv}")


algorithm = RNSGA2(
    ref_points=ref_points,
    pop_size=40,
    epsilon=0.01,
    normalization='front',
    extreme_points_as_reference_points=False,
    weights=np.array([0.5, 0.5]))

res = minimize(problem,
               algorithm,
               save_history=True,
               callback=MyCallback(),
               termination=('n_gen', 300),
               seed=1,
               pf=pf,
               verbose=False)

rmetric = RMetric(problem, ref_points,  delta=0.2)
rigd, rhv = rmetric.do(res.F, others=None)
print(rigd, rhv)
