import numpy as np
from pymoo.model.problem import Problem


class SubsetProblem(Problem):
    def __init__(self,
                 L,
                 n_max
                 ):
        super().__init__(n_var=len(L), n_obj=1, n_constr=1, elementwise_evaluation=True)
        self.L = L
        self.n_max = n_max

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum(self.L[x])
        out["G"] = (self.n_max - np.sum(x)) ** 2


# create the actual problem to be solved
np.random.seed(1)
L = np.array([np.random.randint(100) for _ in range(100)])
n_max = 10
problem = SubsetProblem(L, n_max)

from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.sampling import Sampling


class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=np.bool)

        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[:problem.n_max]
            X[k, I] = True

        return X


class BinaryCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(both_are_true)

            I = np.where(np.logical_xor(p1, p2))[0]

            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True

        return _X


class MyMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            X[i, np.random.choice(is_false)] = True
            X[i, np.random.choice(is_true)] = False

        return X


from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.optimize import minimize

algorithm = GA(
    pop_size=100,
    sampling=MySampling(),
    crossover=BinaryCrossover(),
    mutation=MyMutation(),
    eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               ('n_gen', 75),
               seed=1,
               verbose=True)

print("Function value: %s" % res.F[0])
print("Subset:", np.where(res.X)[0])

opt = np.sort(np.argsort(L)[:n_max])
print("Optimal Subset:", opt)
print("Optimal Function Value: %s" % L[opt].sum())
