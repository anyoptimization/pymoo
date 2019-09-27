import string

import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


class MyProblem(Problem):
    def __init__(self, n_characters=10):
        super().__init__(n_var=1, n_obj=2, n_constr=0, elementwise_evaluation=True)
        self.n_characters = n_characters
        self.ALPHABET = [c for c in string.ascii_lowercase]

    def _evaluate(self, x, out, *args, **kwargs):
        n_a, n_b = 0, 0
        for c in x[0]:
            if c == 'a':
                n_a += 1
            elif c == 'b':
                n_b += 1

        out["F"] = np.array([- n_a, - n_b], dtype=np.float)


class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1), None, dtype=np.object)

        for i in range(n_samples):
            X[i, 0] = "".join([np.random.choice(problem.ALPHABET) for _ in range(problem.n_characters)])

        return X


class MyCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.full_like(X, None, dtype=np.object)

        for k in range(n_matings):
            a, b = X[0, k], X[1, k]
            off_a, off_b = np.full(problem.n_characters, "_"), np.full(problem.n_characters, "_")

            rand = np.random.random(problem.n_characters)

            off_a[rand < 0.5] = a
            off_a[rand >= 0.5] = b

            off_b[rand < 0.5] = b
            off_b[rand >= 0.5] = a

            Y[0, k, 0], Y[1, k, 0] = "".join(off_a), "".join(off_b)

        return Y


class MyMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            if np.random.random() < 0.5:
                mut = [c if np.random.random() > 1 / problem.n_characters else np.random.choice(problem.ALPHABET) for c
                       in X[i, 0]]
                X[i, 0] = "".join(mut)

        return X


def func_is_duplicate(pop, *other, epsilon=1e-20, **kwargs):
    if len(other) == 0:
        return np.full(len(pop), False)

    # value to finally return
    is_duplicate = np.full(len(pop), False)

    H = set()
    for e in other:
        for val in e:
            H.add(val.X[0])

    for i, (val, ) in enumerate(pop.get("X")):
        if val in H:
            is_duplicate[i] = True
        H.add(val)

    return is_duplicate


algorithm = NSGA2(pop_size=100,
                  sampling=MySampling(),
                  crossover=MyCrossover(),
                  mutation=MyMutation(),
                  eliminate_duplicates=func_is_duplicate)

res = minimize(MyProblem(),
               algorithm,
               ('n_gen', 500),
               seed=1,
               verbose=True)

Scatter().add(res.F).show()
print(res.X[np.argsort(res.F[:, 0])])
