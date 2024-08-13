from pymoo.algorithms.moo.pinsga2 import PINSGA2
from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT1
from pymoo.problems.multi import ZDT3
import pymoo.gradient.toolbox as anp
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter
from pymoo.util import value_functions as mvf


# Additional example from original PI-EMO-VF literature
class ZDT1_max(ZDT1):

    def _evaluate(self, x, out, *args, **kwargs): 
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * anp.sum(x[:, 1:], axis=1)
        f2 = (10 - anp.power((f1 * g), 0.5)) / g

        f1 = -1 * f1;
        f2 = -1 * f2;

        out["F"] = anp.column_stack([f1, f2])



def plot_eta_F(context, algorithm):

   
    # Next highlight the eta selctions
    if len(algorithm.eta_F) > 0:

        # Plot the historical PO fronts
        plot = Scatter().add(algorithm.historical_F, facecolors= '#f5f5f5', edgecolors='#f5f5f5')

        # The current PO front 
        plot.add(algorithm.paused_F)

        # Starred items for the DM
        plot.add(algorithm.eta_F, s=500, marker='*', facecolors='red')
        

    else: 
        F = algorithm.pop.get("F")
        plot = Scatter().add(F)


    plot.plot_if_not_done_yet()

    return plot.fig
 

def plot_vf(context, algorithm):

    # This option prevents us from plotting vf every single 
    if not algorithm.vf_plot_flag and algorithm.vf_plot: 
        return algorithm.vf_plot

    # This option is if a vf has been calculated and we want to plot it
    elif len(algorithm.eta_F) > 0 and (algorithm.vf_res is not None) and algorithm.vf_plot_flag:
        plot = mvf.plot_vf(algorithm.eta_F * -1, algorithm.vf_res.vf, show=False)
        
        algorithm.vf_plot_flag = False;
        
        algorithm.vf_plot = plot.gcf()

        return plot.gcf()
   
    # This option is if we have not calculated a vf and we just want to show the PF
    else: 
        F = algorithm.pop.get("F")

        plt.scatter(F[:, 0], F[:, 1], color='blue')
        
        algorithm.vf_plot = plt.gcf()

        return algorithm.vf_plot


problem = ZDT3()

# opt_method can be trust-constr, SLSQP, ES, or GA
# vf_type can be linear or poly
algorithm = PINSGA2(pop_size=30, opt_method="trust-constr", vf_type="poly")

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)

