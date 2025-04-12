import matplotlib.pyplot as plt

from pymoo.core.callback import Callback
from pymoo.visualization.scatter import Scatter


class AnimationCallback(Callback):

    def __init__(self,
                 do_show=False,
                 do_close=True,
                 nth_gen=1,
                 dpi=None,
                 recorder=None,
                 fname=None,
                 exception_if_not_applicable=True):

        super().__init__()
        self.nth_gen = nth_gen
        self.do_show = do_show
        self.do_close = do_close
        self.exception_if_not_applicable = exception_if_not_applicable

        self.recorder = recorder
        if self.recorder is None and fname is not None:
            try:
                from pyrecorder.recorder import Recorder
                from pyrecorder.writers.video import Video
                from pyrecorder.converters.matplotlib import Matplotlib
                self.recorder = Recorder(Video(fname), converter=Matplotlib(dpi=dpi))
            except:
                raise Exception("Please install or update pyrecorder for animation support: pip install -U pyrecorder")

    def update(self, algorithm):
        if algorithm.n_gen == 1 or algorithm.n_gen % self.nth_gen == 0:
            try:

                figure = self.do(algorithm.problem, algorithm)

                if self.do_show:
                    if figure is not None:
                        figure.show()
                    else:
                        plt.show()

                if self.recorder is not None:
                    self.recorder.record(fig=figure)

                if self.do_close:
                    plt.close(fig=figure)

                return figure

            except Exception as ex:
                if self.exception_if_not_applicable:
                    raise ex

    def do(self, problem, algorithm, **kwargs):
        pass


class ObjectiveSpaceAnimation(AnimationCallback):

    def __init__(self, recorder=None, **kwargs):
        if recorder is None:
            from pyrecorder.recorder import Recorder
            from pyrecorder.writers.streamer import Streamer
            recorder = Recorder(Streamer())
        super().__init__(recorder=recorder, **kwargs)

    def update(self, algorithm):
        F = algorithm.opt.get("F")
        pf = algorithm.problem.pareto_front()

        sc = Scatter()
        sc.add(F)
        if pf is not None:
            sc.add(pf, plot_type="line", color="black", alpha=0.7)
        sc.do()

        self.recorder.record()

