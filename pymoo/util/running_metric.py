import matplotlib.pyplot as plt
import numpy as np

from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTerminationWithRenormalization
from pymoo.visualization.video.callback_video import AnimationCallback


class RunningMetric(AnimationCallback):

    def __init__(self,
                 delta_gen,
                 n_plots=4,
                 only_if_n_plots=False,
                 key_press=True,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.delta_gen = delta_gen
        self.key_press = key_press
        self.only_if_n_plots = only_if_n_plots
        self.term = MultiObjectiveSpaceToleranceTerminationWithRenormalization(n_last=100000,
                                                                               all_to_current=True,
                                                                               sliding_window=False)
        self.hist = []
        self.n_plots = n_plots

    def do(self, _, algorithm, force_plot=False, **kwargs):
        self.term.do_continue(algorithm)

        metric = self.term.get_metric()
        metrics = self.term.metrics
        tau = len(metrics)

        # if for whatever reason the metric is not written yet
        if metric is None:
            return

        if (tau + 1) % self.delta_gen == 0 or force_plot:

            _delta_f = metric["delta_f"]
            _delta_ideal = [m['delta_ideal'] > 0.005 for m in metrics]
            _delta_nadir = [m['delta_nadir'] > 0.005 for m in metrics]

            if force_plot or not self.only_if_n_plots or (self.only_if_n_plots and len(self.hist) == self.n_plots - 1):

                fig, ax = plt.subplots()

                if self.key_press:
                    def press(event):
                        if event.key == 'q':
                            algorithm.termination.force_termination = True

                    fig.canvas.mpl_connect('key_press_event', press)

                for k, f in self.hist:
                    ax.plot(np.arange(len(f)) + 1, f, label="t=%s" % (k+1), alpha=0.6, linewidth=3)
                ax.plot(np.arange(len(_delta_f)) + 1, _delta_f, label="t=%s (*)" % (tau+1), alpha=0.9, linewidth=3)

                for k in range(len(_delta_ideal)):
                    if _delta_ideal[k] or _delta_nadir[k]:
                        ax.plot([k+1, k+1], [0, _delta_f[k]], color="black", linewidth=0.5, alpha=0.5)
                        ax.plot([k+1], [_delta_f[k]], "o", color="black", alpha=0.5, markersize=2)

                ax.set_yscale("symlog")
                ax.legend()

                ax.set_xlabel("Generation")
                ax.set_ylabel("$\Delta \, f$", rotation=0)

                if self.key_press:
                    plt.draw()
                    plt.waitforbuttonpress()
                    plt.close('all')

            self.hist.append((tau, _delta_f))
            self.hist = self.hist[-(self.n_plots - 1):]

