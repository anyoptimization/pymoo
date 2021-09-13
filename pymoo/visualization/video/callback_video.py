import matplotlib.pyplot as plt

from pymoo.core.callback import Callback


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

    def notify(self, algorithm, **kwargs):
        if algorithm.n_gen == 1 or algorithm.n_gen % self.nth_gen == 0:
            try:

                figure = self.do(algorithm.problem, algorithm, **kwargs)

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
