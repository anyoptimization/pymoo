import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.survival import Survival
from pymoo.optimize import minimize
from pymoo.util.normalization import normalize
from pymop.problem import Problem


class TutorialProblem(Problem):
    def __init__(self):
        super().__init__(n_var=1, n_obj=1, n_constr=0, xl=0, xu=2, type_var=np.double)

    def _evaluate(self, x, f, *args, **kwargs):
        f[:, 0] = - np.abs(np.sin(np.pi * x[:, 0]))

    def _calc_pareto_front(self, *args, **kwargs):
        X = np.linspace(0, 2, num=1000)
        return np.concatenate((X[:, None], self.evaluate(X[:, None], return_constraint_violation=False)), axis=1)


class ClearingFitnessSurvival(Survival):

    def __init__(self, eps=0.05) -> None:
        super().__init__(True)
        self.eps = eps

    def _do(self, pop, n_survive, algorithm=None, **kwargs):
        X, F = pop.get("X", "F")

        problem = algorithm.problem
        _X = normalize(X, problem.xl, problem.xu)
        D = cdist(_X, _X)

        if F.shape[1] != 1:
            raise ValueError("FitnessSurvival can only used for single objective problems!")

        survivors = np.full(len(pop), False)
        _F = np.copy(F)

        while np.sum(survivors) < n_survive:

            s = np.argmin(_F[:, 0])

            if np.isinf(_F[s, 0]):
                _F = np.copy(F)
                _F[survivors] = np.inf
            else:
                survivors[s] = True
                _F[D[s, :] < self.eps] = np.inf

        return pop[survivors]


problem = TutorialProblem()
# plot_problem_surface(problem, 100)


res = minimize(problem,
               method='ga',
               method_args={
                   'pop_size': 20,
                   'survival': ClearingFitnessSurvival(0.02),
               },
               termination=('n_gen', 200),
               save_history=True,
               disp=True)

from pymoo.util.plotting import animate as func_animtate


def callback(ax, *args):
    ax.set_xlim(0, 2)
    ax.set_ylim(-2, 1)


H = np.concatenate([np.hstack((e.pop.get("X"), e.pop.get("F")))[None, :] for e in res.history], axis=0)
func_animtate('%s.mp4' % problem.name(), H, problem, func_iter=callback)

# print(problem)
