import matplotlib.pyplot as plt

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.factory import get_problem
from pymoo.optimize import minimize

from pymoo.util.running_metric import RunningMetric


class MyRunningMetric(RunningMetric):

    def do(self, _, algorithm, force_plot=False, **kwargs):
        ret = super().do(_, algorithm, force_plot, **kwargs)

        if ret:
            plt.savefig(f"test-{algorithm.n_gen}.png")

        plt.close()


problem = get_problem("sphere")

algorithm = GA(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 20),
               seed=1,
               callback=MyRunningMetric(5, do_show=None, do_close=False, key_press=False),
               save_history=True,
               verbose=True)
