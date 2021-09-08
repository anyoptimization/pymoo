import matplotlib.pyplot as plt

from pymoo.core.callback import Callback


class AnimationCallback(Callback):

    def __init__(self,
                 do_show=False,
                 do_close=True,
                 nth_gen=1,
                 dpi=None,
                 video=None,
                 fname=None,
                 exception_if_not_applicable=True):

        super().__init__()
        self.nth_gen = nth_gen
        self.do_show = do_show
        self.do_close = do_close
        self.exception_if_not_applicable = exception_if_not_applicable

        self.video = video
        if self.video is None and fname is not None:
            try:
                from pyrecorder.recorders.file import File
                from pyrecorder.video import Video
                from pyrecorder.converters.matplotlib import Matplotlib
            except:
                raise Exception("Please install pyrecorder for animation support: pip install pyrecorder")

            self.video = Video(File(fname), converter=Matplotlib(dpi=dpi))

    def notify(self, algorithm, **kwargs):
        if algorithm.n_gen == 1 or algorithm.n_gen % self.nth_gen == 0:
            try:

                figure = self.do(algorithm.problem, algorithm, **kwargs)

                if self.do_show:
                    if figure is not None:
                        figure.show()
                    else:
                        plt.show()

                if self.video is not None:
                    self.video.record(fig=figure)

                if self.do_close:
                    plt.close(fig=figure)

                return figure

            except Exception as ex:
                if self.exception_if_not_applicable:
                    raise ex

    def do(self, problem, algorithm, **kwargs):
        pass
