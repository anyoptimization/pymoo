from typing import Generator

import numpy as np

from pymoo.core.algorithm import Algorithm
from pymoo.core.evaluator import Evaluation
from pymoo.core.fitness import sort_by_fitness
from pymoo.core.output import SingleObjectiveOutput, Output
from pymoo.core.solution import SolutionSet, Solution, merge
from pymoo.operators.sampling import Sampling


class G3PCX(Algorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=Sampling(),
                 n_offsprings=2,
                 n_parents=3,
                 family_size=2,
                 eta=0.1,
                 zeta=0.1,
                 output: Output = SingleObjectiveOutput(),
                 **kwargs):

        super().__init__(output=output, **kwargs)
        self.pop_size = pop_size
        self.sampling = sampling
        self.n_offsprings = n_offsprings
        self.n_parents = n_parents
        self.family_size = family_size

        self.eta = eta
        self.zeta = zeta

    def initialize(self) -> Generator[Evaluation, None, SolutionSet]:
        samples = self.sampling.sample(self.problem, self.pop_size)

        sols = yield from self.evaluator.send(samples)
        sols = sort_by_fitness(sols)

        return sols

    def advance(self) -> Generator[Evaluation, Solution | SolutionSet, SolutionSet]:
        vtype = self.problem.vtype

        # how many loops shall be iterated until one iteration has ended
        n_steps = self.pop_size // self.n_offsprings

        sols = self.sols
        pool = np.arange(len(sols))

        for _ in range(n_steps):

            X = np.full((self.n_parents, self.n_offsprings, vtype.size), np.nan)
            X[0] = sols[0].x

            for i in range(self.n_offsprings):
                k = self.random_state.choice(pool[1:], replace=False, size=self.n_parents - 1)
                X[1:, i] = sols[k].get("X")

            Xp = pcx(X, self.eta, self.zeta, 0)
            off = yield from self.evaluator.send(Xp)

            # get a family and merge the new children
            js = self.random_state.choice(pool, size=self.family_size, replace=False)
            family = merge(sols[js], off)
            fittest = sort_by_fitness(family)[:self.family_size]

            # only keep the fittest of the family
            sols[js] = fittest

            print((sols[js].get("F") != fittest.get("F")).sum())

            sols = sort_by_fitness(sols)

        self.sols = sols
        print("FINAL", sols.get("F").min())

        return sols


def pcx(X, eta, zeta, index, eps=1e-32):
    # the number of parents to be considered
    n_parents, n_matings, n_var = X.shape

    # calculate the differences from all parents to index parent
    diff_to_index = X - X[index]
    dist_to_index = np.linalg.norm(diff_to_index, axis=-1)
    dist_to_index = np.maximum(eps, dist_to_index)

    # find the centroid of the parents
    centroid = np.mean(X, axis=0)

    # calculate the difference between the centroid and the k-th parent
    diff_to_centroid = centroid - X[index]

    dist_to_centroid = np.linalg.norm(diff_to_centroid, axis=-1)
    dist_to_centroid = np.maximum(eps, dist_to_centroid)

    # orthogonal directions are computed
    orth_dir = np.zeros_like(dist_to_index)

    for i in range(n_parents):
        if i != index:
            temp1 = (diff_to_index[i] * diff_to_centroid).sum(axis=-1)
            temp2 = temp1 / (dist_to_index[i] * dist_to_centroid)
            temp3 = np.maximum(0.0, 1.0 - temp2 ** 2)
            orth_dir[i] = dist_to_index[i] * (temp3 ** 0.5)

    # this is the avg of the perpendicular distances from other parents to the parent k
    D_not = orth_dir.sum(axis=0) / (n_parents - 1)

    # generating zero-mean normally distributed variables
    sigma = D_not[:, None] * eta
    rnd = np.random.normal(loc=0.0, scale=sigma)

    # implemented just like the c code - generate_new.h file
    inner_prod = np.sum(rnd * diff_to_centroid, axis=-1, keepdims=True)
    noise = rnd - (inner_prod * diff_to_centroid) / dist_to_centroid[:, None] ** 2

    bias_to_centroid = np.random.normal(0.0, zeta) * diff_to_centroid

    # the array which is finally returned
    Xp = X[index] + noise + bias_to_centroid

    return Xp
