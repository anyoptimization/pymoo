import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.core.callback import CallbackCollection, Callback
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.problems.dyn import TimeSimulation
from pymoo.termination import get_termination
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import Hypervolume
from statistics import mean
from pymoo.algorithms.moo.kgb import KGB

# Experimental Settings
n_var = 5
change_frequency = 10
change_severity = 1
pop_size = 100
max_n_gen = 30 * change_frequency
termination = get_termination("n_gen", max_n_gen)
problem_string = "df1"
verbose = False
seed = 1

# Metric Vars / Callbacks
po_gen = []
igds = []
hvs = []
pof = []
pos = []

def reset_metrics():
    global po_gen, igds, hvs, igds_monitor, hvs_monitor, pof, pos
    po_gen = []
    igds = []
    hvs = []
    igds_monitor = []
    hvs_monitor = []
    pof = []
    pos = []

def update_metrics(algorithm):

    _F = algorithm.opt.get("F")
    PF = algorithm.problem._calc_pareto_front()
    igd = IGD(PF).do(_F)
    hv = Hypervolume(pf=PF).do(_F)

    pos.append(algorithm.opt.get("X"))
    igds.append(igd)
    hvs.append(hv)

    po_gen.append(algorithm.opt)

    pof.append(PF)

class DefaultDynCallback(Callback):

    def _update(self, algorithm):

        update_metrics(algorithm)

# Function to run an algorithm and return the results
def run_algorithm(problem, algorithm, termination, seed, verbose):
    reset_metrics()
    simulation = TimeSimulation()
    callback = CallbackCollection(DefaultDynCallback(), simulation)
    res = minimize(problem, algorithm, termination=termination, callback=callback, seed=seed, verbose=verbose)
    return res, igds, hvs

# Function to plot metrics on an axis
def plot_metrics(ax, data, ylabel, label=None):
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.plot(data, label=label)


def main():
    # DNSGA2
    problem = get_problem(problem_string, taut=change_frequency, nt=change_severity, n_var=n_var)
    algorithm = DNSGA2(pop_size=pop_size)
    start = time.time()
    res, igds, hvs = run_algorithm(problem, algorithm, termination, seed, verbose)
    print("DNSGA2 Performance")
    print(f'Time: {time.time() - start}')
    print("MIGDS", mean(igds))
    print("MHV", mean(hvs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_metrics(ax1, hvs, "Hypervolume", label="DNSGA2")
    plot_metrics(ax2, igds, "IGD", label="DNSGA2")

    # KGB-DMOEA
    problem = get_problem(problem_string, taut=change_frequency, nt=change_severity, n_var=n_var)
    algorithm = KGB(pop_size=pop_size, save_ps=True)
    start = time.time()
    res, igds, hvs = run_algorithm(problem, algorithm, termination, seed, verbose)

    print("KGBDMOEA Performance")
    print(f'Time: {time.time() - start}')
    print("MIGDS", mean(igds))
    print("MHV", mean(hvs))

    plot_metrics(ax1, hvs, "Hypervolume", label="KGB-DMOEA")
    plot_metrics(ax2, igds, "IGD", label="KGB-DMOEA")

    # KGB-DMOA with PS Init load archive of POS

    with open('ps.json', 'r') as f:
        ps = json.load(f)

    problem = get_problem(problem_string, taut=change_frequency, nt=change_severity, n_var=n_var)
    algorithm = KGB(pop_size=pop_size, ps=ps, save_ps=True)
    start = time.time()
    res, igds, hvs = run_algorithm(problem, algorithm, termination, seed, verbose)

    print("KGBDMOEA Performance")
    print(f'Time: {time.time() - start}')
    print("MIGDS", mean(igds))
    print("MHV", mean(hvs))

    plot_metrics(ax1, hvs, "Hypervolume", label="KGB-DMOA with PS Init")
    plot_metrics(ax2, igds, "IGD", label="KGB-DMOA with PS Init")

    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    plt.savefig('output_plot.png')

if __name__ == '__main__':
    main()