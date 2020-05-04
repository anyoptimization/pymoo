import matplotlib.pyplot as plt
import numpy as np
from pyrecorder.recorders.file import File
from pyrecorder.video import Video

from pymoo.algorithms.so_cmaes import CMAES
from pymoo.algorithms.so_nelder_mead import NelderMead
from pymoo.docs import parse_doc_string
from pymoo.model.algorithm import Algorithm
from pymoo.model.individual import Individual
from pymoo.model.initialization import Initialization
from pymoo.model.population import Population
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.problems.single import Rastrigin
from pymoo.util.clearing import select_by_clearing, func_select_by_objective
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.misc import norm_eucl_dist
from pymoo.util.roulette import RouletteWheelSelection
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from pymoo.visualization.fitness_landscape import FitnessLandscape


# =========================================================================================================
# Implementation
# =========================================================================================================


class MMGA(Algorithm):

    def __init__(self,
                 pop_size=200,
                 n_parallel=10,
                 sampling=LatinHypercubeSampling(),
                 display=SingleObjectiveDisplay(),
                 repair=None,
                 individual=Individual(),
                 **kwargs):
        """

        Parameters
        ----------
        pop_size : {pop_size}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """

        super().__init__(display=display, **kwargs)

        self.initialization = Initialization(sampling,
                                             individual=individual,
                                             repair=repair)

        self.pop_size = pop_size
        self.n_parallel = n_parallel
        self.each_pop_size = pop_size // n_parallel

        self.solvers = None
        self.niches = []

        def cmaes(problem, x):
            solver = CMAES(x0=x,
                           tolfun=1e-11,
                           tolx=1e-3,
                           restarts=0)
            solver.initialize(problem)
            solver.next()
            return solver

        def nelder_mead(problem, x):
            solver = NelderMead(X=x)
            solver.initialize(problem)
            solver._initialize()
            solver.n_gen = 1
            solver.next()
            return solver

        self.func_create_solver = nelder_mead

        self.default_termination = SingleObjectiveDefaultTermination()

    def _initialize(self):
        self.pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        self.evaluator.eval(self.problem, self.pop, algorithm=self)

        X = self.pop.get("X")
        D = norm_eucl_dist(self.problem, X, X)
        S = select_by_clearing(self.pop, D, self.n_parallel, func_select_by_objective)

        self.solvers = []
        for s in S:
            solver = self.func_create_solver(self.problem, self.pop[s].X)
            self.solvers.append(solver)

    def _next(self):
        n_evals = np.array([solver.evaluator.n_eval for solver in self.solvers])
        ranks = np.array([solver.opt[0].F[0] for solver in self.solvers]).argsort() + 1

        rws = RouletteWheelSelection(ranks, larger_is_better=False)
        S = rws.next()
        self.solvers[S].next()

        print(n_evals.sum(), n_evals)

        if self.solvers[S].termination.force_termination or self.solvers[S].termination.has_terminated(self.solvers[S]):
            self.niches.append(self.solvers[S])
            print(self.solvers[S].opt.get("F"), self.solvers[S].opt.get("X"))
            self.solvers[S] = None

        for k in range(self.n_parallel):
            if self.solvers[k] is None:
                x = FloatRandomSampling().do(self.problem, 1)[0].get("X")
                self.solvers[S] = self.func_create_solver(self.problem, x)

    def _set_optimum(self, force=False):
        self.opt = Population()
        for solver in self.niches:
            self.opt = Population.merge(self.opt, solver.opt)


parse_doc_string(MMGA.__init__)

if __name__ == '__main__':
    problem = Rastrigin()

    algorithm = MMGA()

    ret = minimize(problem,
                   algorithm,
                   termination=('n_gen', 10000),
                   seed=1,
                   save_history=True,
                   verbose=False)

    print(ret.F)

    with Video(File("mm.mp4")) as vid:
        for algorithm in ret.history:

            if algorithm.n_gen % 100 == 0:

                fl = FitnessLandscape(problem,
                                      _type="contour",
                                      kwargs_contour=dict(linestyles="solid", offset=-1, alpha=0.4))
                fl.do()

                for k, solver in enumerate(algorithm.solvers):
                    X = solver.pop.get("X")
                    plt.scatter(X[:, 0], X[:, 1], marker="x", label=k)

                for k, solver in enumerate(algorithm.niches):
                    X = solver.opt.get("X")
                    plt.scatter(X[:, 0], X[:, 1], marker="x", color="black", label="Niche %s" % k, s=80)

                plt.legend()
                plt.title(algorithm.n_gen)

                # sc = Scatter(title=algorithm.n_gen)
                # sc.add(curve(algorithm.problem), plot_type="line", color="black")
                # sc.add(np.column_stack([pop.get("X"), pop.get("F")]), color="red")
                # sc.do()

                vid.record()

    # with Video(File("mm.mp4")) as vid:
    #     for algorithm in ret.history:
    #         pop = algorithm.pop
    #
    #         sc = Scatter(title=algorithm.n_gen)
    #         sc.add(curve(algorithm.problem), plot_type="line", color="black")
    #         sc.add(np.column_stack([pop.get("X"), pop.get("F")]), color="red")
    #         sc.do()
    #
    #         vid.record()
