from pymoo.core.callback import Callback
from pymoo.util.display.progress import ProgressBar


class Display(Callback):

    def __init__(self, output=None, progress=False, verbose=False):
        super().__init__()
        self.output = output
        self.verbose = verbose
        self.progress = ProgressBar() if progress else None

    def update(self, algorithm, **kwargs):
        output, progress = self.output, self.progress

        if self.verbose and output:
            text = ""
            header = not output.is_initialized
            output(algorithm)

            if header:
                text += output.header(border=True) + '\n'
            text += output.text()

            print(text)

        if progress:
            perc = algorithm.termination.perc
            progress.set(perc)

    def finalize(self):

        if self.progress:
            self.progress.close()
