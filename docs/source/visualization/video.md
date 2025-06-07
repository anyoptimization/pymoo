---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

```{raw-cell}
---
pycharm:
  name: '#%% raw

    '
raw_mimetype: text/restructuredtext
---
.. _nb_video:
```

+++ {"pycharm": {"name": "#%% md\n"}}

## Video

+++ {"pycharm": {"name": "#%% md\n"}}

Images are fantastic, but since optimization happens over time, videos can often capture the optimization process itself in a better way.
In `pymoo` we offer a wrapper around `matplotlib` to combine complex plots and put them together in a video (a different way than the animation package does it). This is, however, more computationally expensive, but makes recording very simple.

+++ {"pycharm": {"name": "#%% md\n"}}

To enable video support, you have to install `pyrecorder` by

```
pip install -U pyrecorder
```

Because our recording tool has some dependencies that not every regular `pymoo` user would be interested in, we have decided to outsource the recording to another third-party library.

+++ {"pycharm": {"name": "#%% md\n"}}

For instance, let us record a short video with only three frames (randomly created scatter plots):

```{code-cell}
---
pycharm:
  name: '#%%

    '
---
import numpy as np
from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter


with Recorder(Video("example.mp4")) as rec:

    for k in range(10):
        X = np.random.random((100, 2))
        Scatter(title=str(k)).add(X).do()
        rec.record()

```

+++ {"pycharm": {"name": "#%% md\n"}}

Or recording a video after the run has finished:

```{code-cell}
---
pycharm:
  name: '#%%

    '
---
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("zdt1")

algorithm = NSGA2(pop_size=100, eliminate_duplicates=True)

ret = minimize(problem,
               algorithm,
               termination=('n_gen', 100),
               seed=1,
               save_history=True,
               verbose=False)

# use the video writer as a resource
with Recorder(Video("ga.mp4")) as rec:

    # for each algorithm object in the history
    for entry in ret.history:
        sc = Scatter(title=("Gen %s" % entry.n_gen))
        sc.add(entry.pop.get("F"))
        sc.add(entry.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
        sc.do()

        # finally record the current visualization to the video
        rec.record()
```

+++ {"pycharm": {"name": "#%% md\n"}}

The callback directive can be used to initiate a video by `Video(Streamer())` to stream the current algorithm state directly to the screen.
This allows the streaming of the current status of the algorithm.

```{code-cell}
---
pycharm:
  name: '#%%

    '
---
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.pcp import PCP
from pyrecorder.recorder import Recorder
from pyrecorder.writers.streamer import Streamer


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.rec = Recorder(Streamer(sleep=0.1))

    def notify(self, algorithm):
        problem = algorithm.problem

        pcp = PCP(title=("Gen %s" % algorithm.n_gen, {'pad': 30}),
                  bounds=(problem.xl, problem.xu),
                  labels=["$x_%s$" % k for k in range(problem.n_var)]
                  )
        pcp.set_axis_style(color="grey", alpha=0.5)

        pcp.add(algorithm.pop.get("X"), color="black", alpha=0.8, linewidth=1.5)
        if algorithm.off is not None:
            pcp.add(algorithm.off.get("X"), color="blue", alpha=0.8, linewidth=0.5)

        pcp.add(algorithm.opt.get("X"), color="red", linewidth=4)
        pcp.do()

        self.rec.record()


problem = get_problem("rastrigin", n_var=10)

algorithm = GA(pop_size=50, eliminate_duplicates=True, callback=MyCallback())

ret = minimize(problem,
               algorithm,
               termination=('n_gen', 50),
               seed=1,
               verbose=False)

```
