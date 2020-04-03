import matplotlib.pyplot as plt
import numpy as np

from pymoo.model.callback import Callback
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTerminationWithRenormalization


class RunningMetric(Callback):

    def __init__(self, nth_gen, n_plots=4) -> None:
        super().__init__()
        self.nth_gen = nth_gen
        self.term = MultiObjectiveSpaceToleranceTerminationWithRenormalization(n_last=30,
                                                                               all_to_current=True,
                                                                               sliding_window=False)

        self.hist = []
        self.n_hist = n_plots

    def notify(self, algorithm):
        self.term.do_continue(algorithm)
        t = algorithm.n_gen

        def press(event):
            if event.key == 'q':
                algorithm.termination.force_termination = True

        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', press)

        metric = self.term.get_metric()
        metrics = self.term.metrics
        tau = len(metrics)

        if metric is not None and tau % self.nth_gen == 0:

            for k, f in self.hist:
                plt.plot(np.arange(len(f)), f, label="t=%s" % k, alpha=0.6, linewidth=3)

            _delta_f = metric["delta_f"]
            plt.plot(np.arange(len(_delta_f)), _delta_f, label="t=%s (*)" % tau, alpha=0.9, linewidth=3)

            _delta_ideal = [m['delta_ideal'] > 0.005 for m in metrics]
            _delta_nadir = [m['delta_nadir'] > 0.005 for m in metrics]

            for k in range(len(_delta_ideal)):
                if _delta_ideal[k] or _delta_nadir[k]:
                    plt.plot([k, k], [0, _delta_f[k]], color="black", linewidth=0.5, alpha=0.5)
                    plt.plot([k], [_delta_f[k]], "o", color="black", alpha=0.5, markersize=2)

            self.hist.append((tau, _delta_f))
            if self.n_hist is not None:
                self.hist = self.hist[-(self.n_hist - 1):]

            plt.yscale("symlog")
            plt.legend()

            plt.xlabel("Generation")
            plt.ylabel("$\Delta \, f$", rotation=0)

            plt.draw()
            plt.waitforbuttonpress()

            fig.clf()
            plt.close(fig)
