import numpy as np

from pymoo.algorithms.so_nelder_mead import NelderAndMeadTermination
from pymoo.factory import get_algorithm, normalize
from pymoo.model.algorithm import Algorithm
from pymoo.model.evaluator import Evaluator
from pymoo.model.population import Population, pop_from_array_or_individual
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.util.display import disp_single_objective
from pymoo.util.misc import pop_from_sampling, evaluate_if_not_done_yet, vectorized_cdist, norm_euclidean_distance

# =========================================================================================================
# Implementation
# =========================================================================================================
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class GlobalEvaluator(Evaluator):

    def __init__(self):
        super().__init__()
        self.algorithms = {}
        self.history = None
        self.opt = None

    def eval(self, problem, X, algorithm=None, **kwargs):

        pop = pop_from_array_or_individual(X)
        pop.set("algorithm", [algorithm] * len(pop))

        if self.history is None:
            self.history = pop
        else:
            self.history = Population.create(self.history, pop)

        before = self.n_eval
        ret = super().eval(problem, X, **kwargs)
        after = self.n_eval

        if algorithm is not None:
            if algorithm not in self.algorithms:
                self.algorithms[algorithm] = 0

            self.algorithms[algorithm] += after - before

        if self.opt is None or pop.get("F").min() < self.opt[0].F.min():
            self.opt = pop[[pop.get("F").argmin()]]

        return ret


def predict_by_nearest_neighbors(X, F, X_pred, n_nearest, problem):
    D = vectorized_cdist(X_pred, X, func_dist=norm_euclidean_distance(problem))
    nearest = np.argsort(D, axis=1)[:, :n_nearest]

    I = np.arange(len(D))[None, :].repeat(n_nearest, axis=0).T
    dist_to_nearest = D[I, nearest]

    w = dist_to_nearest / dist_to_nearest.sum(axis=1)[:, None]

    F_pred = (F[:, 0][nearest] * w).sum(axis=1)
    F_uncert = dist_to_nearest.mean(axis=1)

    return F_pred, F_uncert


class SingleObjectiveGlobalOptimization(Algorithm):

    def __init__(self,
                 n_initial_samples=50,
                 n_parallel_searches=5,
                 **kwargs):
        super().__init__(**kwargs)

        # the algorithm to be used for optimization
        self.n_parallel_searches = n_parallel_searches

        # the initial sampling to be used
        self.sampling = LatinHypercubeSampling(iterations=100)
        self.n_initial_samples = n_initial_samples

        # create a global evaluator that keeps track if the inner algorithms as well
        self.evaluator = GlobalEvaluator()

        # objects used during the optimization
        self.algorithms = []

        # display the single-objective metrics
        self.func_display_attrs = disp_single_objective

    def _initialize(self):
        pop = pop_from_sampling(self.problem, self.sampling, self.n_initial_samples)
        evaluate_if_not_done_yet(self.evaluator, self.problem, pop, algorithm=self)

        for i in np.argsort(pop.get("F")[:, 0]):
            algorithm = get_algorithm("nelder-mead",
                                      problem=self.problem,
                                      x0=pop[i],
                                      termination=NelderAndMeadTermination(x_tol=1e-3, f_tol=1e-3),
                                      evaluator=self.evaluator
                                      )
            algorithm.initialize()
            self.algorithms.append(algorithm)

        self.pop = pop

    def _next(self):

        # all place visited so far
        _X, _F, _evaluated_by_algorithm = self.evaluator.history.get("X", "F", "algorithm")

        # collect attributes from each algorithm and determine whether it has to be replaced or not
        pop, F, n_evals = [], [], []
        for k, algorithm in enumerate(self.algorithms):

            # collect some data from the current algorithms
            _pop = algorithm.pop

            # if the algorithm has terminated or not
            has_finished = algorithm.termination.has_terminated(algorithm)

            # if the area was already explored before
            closest_dist_to_others = vectorized_cdist(_pop.get("X"), _X[_evaluated_by_algorithm != algorithm],
                                                      func_dist=norm_euclidean_distance(self.problem))
            too_close_to_others = (closest_dist_to_others.min(axis=1) < 1e-3).all()

            # whether the algorithm is the current best - if yes it will not be replaced
            current_best = self.evaluator.opt.get("F") == _pop.get("F").min()

            # algorithm not really useful anymore
            if not current_best and (has_finished or too_close_to_others):
                # find a suitable x0 which is far from other or has good expectations
                self.sampling.criterion = lambda X: vectorized_cdist(X, _X).min()
                X = self.sampling.do(self.problem, self.n_initial_samples).get("X")

                # distance in x space to other existing points
                x_dist = vectorized_cdist(X, _X, func_dist=norm_euclidean_distance(self.problem)).min(axis=1)
                f_pred, f_uncert = predict_by_nearest_neighbors(_X, _F, X, 5, self.problem)
                fronts = NonDominatedSorting().do(np.column_stack([- x_dist, f_pred, f_uncert]))
                I = np.random.choice(fronts[0])

                # I = vectorized_cdist(X, _X, func_dist=norm_euclidean_distance(self.problem)).min(axis=1).argmax()

                # choose the one with the largest distance to current solutions
                x0 = X[[I]]

                # replace the current algorithm
                algorithm = get_algorithm("nelder-mead",
                                          problem=self.problem,
                                          x0=x0,
                                          termination=NelderAndMeadTermination(x_tol=1e-3, f_tol=1e-3),
                                          evaluator=self.evaluator,
                                          )
                algorithm.initialize()
                self.algorithms[k] = algorithm

            pop.append(algorithm.pop)
            F.append(algorithm.pop.get("F"))
            n_evals.append(self.evaluator.algorithms[algorithm])

        # get the values of all algorithms as arrays
        F, n_evals = np.array(F), np.array(n_evals)
        rewards = 1 - normalize(F.min(axis=1))[:, 0]
        n_evals_total = self.evaluator.n_eval - self.evaluator.algorithms[self]

        # calculate the upper confidence bound
        ucb = rewards + 0.95 * np.sqrt(np.log(n_evals_total) / n_evals)

        I = ucb.argmax()
        self.algorithms[I].next()

        # create the population object with all algorithms
        self.pop = Population.create(*pop)

        # update the current optimum
        self.opt = self.evaluator.opt

        #print(n_evals)


# =========================================================================================================
# Interface
# =========================================================================================================

def global_optimization(**kwargs):
    return SingleObjectiveGlobalOptimization(**kwargs)
