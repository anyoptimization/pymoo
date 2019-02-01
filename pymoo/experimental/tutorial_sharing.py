import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.survival import Survival
from pymoo.optimize import minimize
from pymop import plot_problem_surface
from pymop.problem import Problem


class TutorialProblem(Problem):
    def __init__(self):
        super().__init__(n_var=1, n_obj=1, n_constr=0, xl=0, xu=2, type_var=np.double)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.abs(np.sin(np.pi * x[:, 0]))

    def _calc_pareto_front(self, *args, **kwargs):
        X = np.linspace(0, 2, num=1000)
        return np.concatenate((X[:, None], self.evaluate(X[:, None], return_values_of=["F"])), axis=1)


class TutorialProblem2(Problem):
    def __init__(self):
        super().__init__(n_var=1, n_obj=1, n_constr=0, xl=0, xu=1, type_var=np.double)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = 1.1 - np.exp(-2*x[:,0]) * np.power(np.sin(5 * np.pi * x[:,0]), 2)

    def _calc_pareto_front(self, *args, **kwargs):
        X = np.linspace(0, 2, num=1000)
        return np.concatenate((X[:, None], self.evaluate(X[:, None], return_values_of=["F"])), axis=1)


class FitnessSharingSurvival(Survival):

    def __init__(self, sigma=0.5, alpha=1) -> None:
        """

        Parameters
        ----------
        sigma : double
            (max - min) / 2*number_of_optima

        alpha : double
            default: 1

        """
        super().__init__(True)
        self.alpha = alpha
        self.sigma = sigma

    def _do(self, pop, n_survive, algorithm=None, **kwargs):
        X, F = pop.get("X", "F")

        if F.shape[1] != 1:
            raise ValueError("FitnessSurvival can only used for single objective problems!")

        # normalized distance in the design space
        #problem = algorithm.problem
        #_X = normalize(X, problem.xl, problem.xu)
        _X = X
        D = cdist(_X, _X)

        # calculate the niche count
        nc = 1 - (np.power(D / self.sigma, self.alpha))
        nc[D > self.sigma] = 0
        nc = np.sum(nc, axis=1)

        # modified objective value
        # ???????
        _F = F[:, 0] / nc
        #_F = (F[:,0] + np.min(F)) * nc

        return pop[np.argsort(_F)[:n_survive]]


problem = TutorialProblem()
plot_problem_surface(problem, 100)

X = np.array([[1.651], [1.397], [0.921], [0.349], [1.524], [1.460]])
F, _ = problem.evaluate(X)
print(np.round(F,3))

res = minimize(problem,
               method='ga',
               method_args={
                   'pop_size': 10,
                   'sampling': X,
                   'survival': FitnessSharingSurvival(sigma=0.5, alpha=1.0),
               },
               termination=('n_gen', 100),
               save_history=True,
               disp=True)

from pymoo.util.plotting import animate as func_animtate


def callback(ax, *args):

    # 1
    #ax.set_xlim(0, 2)
    #ax.set_ylim(-1.2, 0.1)

    #2
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1.2)


H = np.concatenate([np.column_stack(e.pop.get("X", "F"))[None, :] for e in res.history], axis=0)
func_animtate('%s.mp4' % problem.name(), H, problem, func_iter=callback)

# print(problem)
