from pymoo.visualization.Dashboard import Dashboard
from pymoo.algorithms.moo.pinsga2 import PINSGA2
from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT1
from pymoo.problems.multi import ZDT3
import pymoo.gradient.toolbox as anp
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter
from pymoo.util import value_functions as mvf

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
problem = ZDT3()

algorithm = PINSGA2(pop_size=30)

def plot_eta_F(context, algorithm):

   
    # Next highlight the eta selctions
    if len(algorithm.eta_F) > 0:

        # Plot the historical PO fronts
        plot = Scatter().add(algorithm.historical_F * -1, facecolors= '#f5f5f5', edgecolors='#f5f5f5')

        # The current PO front 
        plot.add(algorithm.paused_F * -1)

        # Starred items for the DM
        plot.add(algorithm.eta_F * -1, s=500, marker='*', facecolors='red')
        

    else: 
        F = algorithm.pop.get("F")
        plot = Scatter().add(F * -1)


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


def plot_dom(context, algorithm):

    F = algorithm.pop.get("F")

    fronts = algorithm.fronts

    
    if len(algorithm.eta_F) > 0 and (algorithm.vf_res is not None):
        plot = mvf.plot_vf(algorithm.eta_F * -1, algorithm.vf_res.vf, show=False)
    else:
        plot = plt
    


    if len(fronts) == 0: 
        plot.scatter(-1*F[:, 0], -1*F[:, 1], color="blue")

    else: 

        # TODO make this more automatic
        colors = ['red', 'blue', 'yellow', 'green', 'purple', 'orange', 'black', 'pink', 'brown', 'gray']

        # Plot each of the fronts
        for f in range(0, max(fronts)):
            plot.scatter(-1*F[fronts == f, 0], -1*F[fronts == f, 1], 
                        color=colors[f], 
                        label="Front %d" % (f + 1))

         

        plot.legend()

    ax = plot.gca()

    ax.set_xlim([min(-1*F[:, 0]) * 0.9, max(-1*F[:, 0]) * 1.1])
    ax.set_ylim([min(-1*F[:, 1]) * 0.9, max(-1*F[:, 1]) * 1.1])

    return plot.gcf() 




res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               callback=Dashboard(plot_eta_F=plot_eta_F, plot_vf=plot_vf),
               verbose=True)







