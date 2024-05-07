from pymoo.visualization.Dashboard import Dashboard
from pymoo.algorithms.moo.pinsga2 import PINSGA2
from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT1
import pymoo.gradient.toolbox as anp
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter

class ZDT1_max(ZDT1):

    def _evaluate(self, x, out, *args, **kwargs): 
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * anp.sum(x[:, 1:], axis=1)
        #f2 = g * (1 - anp.power((f1 / g), 0.5))
        f2 = (10 - anp.power((f1 * g), 0.5)) / g

        f1 = -1 * f1;
        f2 = -1 * f2;

        out["F"] = anp.column_stack([f1, f2])




problem = ZDT1_max()

algorithm = PINSGA2(pop_size=100)

def plot_eta_F(context, algorithm):

   
    # Next highlight the eta selctions
    if len(algorithm.eta_F) > 0:
        plot = Scatter().add(algorithm.paused_F)
        plot.add(algorithm.eta_F)

    else: 
        F = algorithm.pop.get("F")
        plot = Scatter().add(F)



    plot.plot_if_not_done_yet()

    return plot.fig
 

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               callback=Dashboard(plot_eta_F=plot_eta_F),
               verbose=True)







