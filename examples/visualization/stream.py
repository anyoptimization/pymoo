from pyrecorder.recorder import Recorder
from pyrecorder.writers.streamer import Streamer

from pymoo.algorithms.soo.nonconvex.ga import GA


from pymoo.problems import get_problem
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.visualization.pcp import PCP


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.rec = Recorder(Streamer())

    def update(self, algorithm):
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
