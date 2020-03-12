import io

try:
    import cv2
except:
    raise Exception("For video support cv2 needs to be installed: pip install opencv-python")
import matplotlib.pyplot as plt
import numpy as np


class Video:

    def __init__(self,
                 fname=None,
                 codec=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                 live=False,
                 hold=True,
                 delay=1,
                 wait_for_key=False,
                 fps=1) -> None:

        super().__init__()
        self.fname = fname

        self.live = live
        self.key = hold

        self.delay = delay
        if wait_for_key:
            self.delay = 0

        self.writer = None
        self.codec = codec
        self.fps = fps

    def snap(self, plt_close=True, dpi=100, duration=1):
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)

        buf.seek(0)
        _bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
        frame = cv2.imdecode(_bytes, cv2.IMREAD_COLOR)

        if self.fname is not None:

            if self.writer is None:
                height, width, layers = frame.shape
                self.writer = cv2.VideoWriter(self.fname, self.codec, self.fps, frameSize=(width, height))

            for k in range(duration):
                self.writer.write(frame)

        if self.live:
            cv2.imshow('image', frame)
            if self.key:
                cv2.waitKey(self.delay)

        if plt_close:
            plt.close()

    def close(self):
        cv2.destroyAllWindows()
        if self.writer is not None:
            self.writer.release()

    @classmethod
    def from_iteratable(cls, fname, iteratable, func, **kwargs):
        vid = Video(fname, **kwargs)

        for entry in iteratable:
            ret = func(entry)
            if not (isinstance(ret, bool) and not ret):
                vid.snap()

        vid.close()


# =========================================================================================================
# Plotting Methods
# =========================================================================================================


def plot_objective_space(algorithm):
    F = algorithm.pop.get("F")
    plt.scatter(F[:, 0], F[:, 1])
    plt.title("Generation %s" % algorithm.n_gen)
