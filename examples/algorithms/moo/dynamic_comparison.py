from copy import copy

import matplotlib.pyplot as plt
import pandas as pd

from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.kgb import KGB
from pymoo.core.callback import CallbackCollection, Callback
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems.dyn import TimeSimulation
from pymoo.problems.dynamic.df import DF1


class DynamicIGD(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data = []

    def _update(self, algorithm):
        pf = algorithm.problem.pareto_front()
        F = algorithm.opt.get("F")
        time = algorithm.problem.time
        igd = IGD(pf).do(F)

        entry = dict(time=time, igd=igd)
        self.data.append(entry)

    def get(self):
        df = pd.DataFrame(self.data)
        migd = df['igd'].mean()
        return migd, df


problem = DF1(taut=2, n_var=2)
n_time = 10

dnsga2 = DNSGA2(version="A")
dnsga2_migd = DynamicIGD()

minimize(copy(problem),
         dnsga2,
         termination=('n_gen', n_time),
         callback=CallbackCollection(dnsga2_migd, TimeSimulation()),
         seed=1,
         verbose=True)

dnsga2_migd, dnsga2_igd_over_time = dnsga2_migd.get()

kgb = KGB()
kgb_migd = DynamicIGD()

minimize(copy(problem),
         kgb,
         termination=('n_gen', n_time),
         callback=CallbackCollection(kgb_migd, TimeSimulation()),
         save_history=True,
         seed=1,
         verbose=True)

kgb_migd, kgb_igd_over_time = kgb_migd.get()

plt.plot(dnsga2_igd_over_time['time'], dnsga2_igd_over_time['igd'], color='black', lw=0.7, label="DNSGA-II")
plt.plot(kgb_igd_over_time['time'], kgb_igd_over_time['igd'], color='red', lw=0.7, label="KGB")
plt.title("Dynamic Optimization")
plt.xlabel("Time")
plt.ylabel("IGD")
plt.legend()
plt.show()
