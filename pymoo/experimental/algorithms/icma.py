import numpy as np
from numpy.random import permutation

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.core.population import Population
from pymoo.core.selection import Selection
from pymoo.operators.crossover.dex import DEX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.normalization import ObjectiveSpaceNormalization
from pymoo.util.optimum import filter_optimum
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination
from pymoo.util.termination.max_gen import MaximumGenerationTermination


def norm(V):
    return V / np.sqrt(np.sum(V ** 2, axis=1))[:, None]


class NeighborhoodSelection(Selection):

    def __init__(self, n_neighbors=10) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors

    def _do(self, pop, n_select, n_parents, ideal=None, **kwargs):
        assert n_parents == 3, "This selection is based on three parents!"
        assert ideal is not None, "The neighborhood selection needs the ideal point for normalization!"

        F = pop.get("F")
        F = F - ideal + 1e-64
        F = norm(F)

        cosv = F @ F.T - 3 * np.eye(len(F))
        I = np.argsort(-cosv, axis=1)
        neighbours = I[:, :self.n_neighbors]

        P = RandomSelection().do(pop, n_select, n_parents)
        P[:, 0] = np.arange(len(P))

        for k in range(len(pop)):

            if np.random.random() < 0.7:
                P[k, 1:] = np.random.choice(neighbours[k], 2, replace=False)
            else:
                P[k, 1:2] = np.random.choice(neighbours[k], 1, replace=False)

        return P


def Indicator_based_CHT(F, indicator, W, N):
    IrFitness = indicator.min(axis=1)
    dom = np.where(IrFitness >= 0)[0]

    if len(dom) <= N:
        return np.argsort(-IrFitness)[:N]
    else:

        F = F[dom]
        indicator = np.copy(indicator[dom][:, dom])

        survivors = np.full(len(F), True)

        # normalize the weights and the objectives
        nW, nF = norm(W), norm(F)

        # assigned each solution based on the angle to a niche
        A = np.argmax((nF @ nW.T), axis=1)

        # create an assignment list for each niche
        niches = [[] for _ in range(len(W))]
        [niches[j].append(i) for i, j in enumerate(A)]

        # the niche counts
        cnt = [len(inds) for inds in niches]

        neighbor = indicator.argmin(axis=1)
        vals = indicator.min(axis=1)

        while survivors.sum() > N:

            # find the niche where a solution should be deleted
            niche = np.argmax(cnt)

            # all solutions in that niche
            I = niches[niche]

            # select the solution in this niche with the worst indicator value
            j = I[vals[I].argmin()]

            # set the values in the indicator matrix to infinity
            indicator[j, :] = np.inf
            indicator[:, j] = np.inf
            vals[j] = np.inf

            # now update the neighbors and matrix partially (only the one that have changed)
            K = np.where(j == neighbor)[0]
            if len(K) > 0:
                neighbor[K] = indicator[K].argmin(axis=1)
                vals[K] = indicator[K].min(axis=1)

            # remove the solution from the niche
            niches[niche] = [e for e in I if e != j]
            cnt[niche] -= 1

            # deselect the solution to survive
            survivors[j] = False

        return dom[survivors]


def Selection_Operator_of_PREA(F, indicator, N):
    fitness = indicator.min(axis=1)
    J = np.where(fitness >= 0)[0]

    if len(J) <= N:
        return np.argsort(-fitness)[:N]
    else:

        survivors = np.full(len(J), True)

        _F = F[J]

        ir = np.copy(indicator[J][:, J])
        ir_neighbor = ir.argmin(axis=1)
        ir_vals = ir.min(axis=1)

        while survivors.sum() > N:

            # select the solution in this niche with the worst indicator value
            j = ir_vals.argmin()

            # set the values in the indicator matrix to infinity
            ir[j] = np.inf
            ir[:, j] = np.inf

            # now update the neighbors and matrix partially (only the one that have changed)
            K = np.where(j == ir_neighbor)[0]
            if len(K) > 0:
                ir_neighbor[K] = ir[K].argmin(axis=1)
                ir_vals[K] = ir[K].min(axis=1)

            # the value of the solution itself becomes infinity
            ir_vals[j] = np.inf

            # deselect the solution to survive
            survivors[j] = False

        nadir = _F[survivors].max(axis=0)

        J = np.where(np.all(F <= nadir, axis=1))[0]

        if len(J) <= N:
            return J

        _F = F[J] / nadir

        survivors = np.full(len(_F), True)

        ir = np.copy(indicator[J][:, J])
        ir_neighbor = ir.argmin(axis=1)
        ir_vals = ir.min(axis=1)

        dist = np.zeros((len(_F), len(_F)))
        for i in range(len(_F)):
            Fi = _F[i]
            Fdelta = _F - Fi
            dist[i, :] = np.sqrt(np.sum(Fdelta ** 2, axis=1) - np.sum(Fdelta, axis=1) ** 2 / _F.shape[1])
            dist[i, i] = np.inf

        dist_neighbor = np.argmin(dist, axis=1)
        dist_vals = np.min(dist, axis=1)

        while survivors.sum() > N:

            a = np.argmin(dist_vals)
            b = dist_neighbor[a]

            if ir_vals[a] < ir_vals[b]:
                j = a
            else:
                j = b

            # set the values in the distance matrix to infinity
            dist[j, :] = np.inf
            dist[:, j] = np.inf

            # now update the neighbors and matrix partially (only the one that have changed)
            K = np.where(j == dist_neighbor)[0]
            if len(K) > 0:
                dist_neighbor[K] = dist[K].argmin(axis=1)
                dist_vals[K] = dist[K].min(axis=1)

            # the value of the solution itself becomes infinity
            dist_vals[j] = np.inf

            # set the values in the indicator matrix to infinity
            ir[j] = np.inf
            ir[:, j] = np.inf

            # now update the neighbors and matrix partially (only the one that have changed)
            K = np.where(j == ir_neighbor)[0]
            if len(K) > 0:
                ir_neighbor[K] = ir[K].argmin(axis=1)
                ir_vals[K] = ir[K].min(axis=1)

            # the value of the solution itself becomes infinity
            ir_vals[j] = np.inf

            # deselect the solution to survive
            survivors[j] = False

        return J[survivors]


class ICMA(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 pop_size=None,
                 sampling=FloatRandomSampling(),
                 selection=NeighborhoodSelection(),
                 crossover=DEX(prob=0.9, CR=0.5, variant='bin'),
                 mutation=PM(eta=20),
                 display=MultiObjectiveDisplay(),
                 **kwargs):

        # set reference directions and pop_size
        self.ref_dirs = ref_dirs
        if self.ref_dirs is not None:
            if pop_size is None:
                pop_size = len(self.ref_dirs)

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         eliminate_duplicates=False,
                         display=display,
                         **kwargs)

        self.RA = None
        self.norm = None
        self.archive = None

    def _setup(self, problem, **kwargs):

        self.norm = ObjectiveSpaceNormalization()

        # if maximum functions termination convert it to generations
        if isinstance(self.termination, MaximumFunctionCallTermination):
            n_gen = np.ceil((self.termination.n_max_evals - self.pop_size) / self.n_offsprings)
            self.termination = MaximumGenerationTermination(n_gen)

        # check whether the n_gen termination is used - otherwise this algorithm can be not run
        if not isinstance(self.termination, MaximumGenerationTermination):
            raise Exception("Please use the n_gen or n_eval as a termination criterion to run ICMA!")

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills, **kwargs)
        self.norm.update(infills)
        self.archive = infills
        self.Ra = 1.0

    def _infill(self, **kwargs):

        pop, archive, Ra = self.pop, self.archive, self.Ra

        N = self.pop_size
        Nt = int(np.floor(Ra * N))

        from_pop = pop[permutation(len(pop))[:Nt]]
        from_archive = pop[permutation(len(archive))[:N - Nt]]
        pool = Population.merge(from_pop, from_archive)

        # first do the differential evolution mating
        off = self.mating.do(self.problem, pool, self.n_offsprings, algorithm=self,
                             ideal=self.norm.ideal(only_feas=False))

        return off

    def _advance(self, infills=None, **kwargs):
        self.norm.update(infills)

        pop = Population.merge(Population.merge(self.pop, infills), self.archive)

        F = pop.get("F")

        if self.problem.has_constraints():
            G = pop.get("G")
            C = np.maximum(0, G)
            C = C / np.maximum(1, C.max(axis=0))
            CV = C.sum(axis=1)
        else:
            CV = np.zeros(len(pop))

        F = F - self.norm.ideal(only_feas=False) + 1e-6

        indicator = np.full((len(pop), len(pop)), np.inf)

        for i in range(len(pop)):
            Ci = CV[i]
            Fi = F[i]

            if Ci == 0:

                Ir = np.log(Fi / F)
                MaxIr = np.max(Ir, axis=1)
                MinIr = np.min(Ir, axis=1)

                CVA = MaxIr

                J = MaxIr <= 0
                CVA[J] = MinIr[J]

                val = CVA

            else:
                IC = (Ci + 1e-6) / (CV + 1e-6)
                CVF = np.max(np.maximum(Fi, F) / np.minimum(Fi, F), axis=1)
                val = np.log(np.maximum(CVF, IC))

            indicator[:, i] = val
            indicator[i, i] = np.inf

        feas = np.where(CV <= 0)[0]

        if len(feas) <= self.pop_size:
            self.archive = pop[np.argsort(CV)[:self.pop_size]]
        else:
            feas_F = pop[feas].get("F") - self.norm.ideal(only_feas=True) + 1e-6
            I = Selection_Operator_of_PREA(feas_F, indicator[feas][:, feas], self.pop_size)
            self.archive = pop[feas[I]]

        I = Indicator_based_CHT(F, indicator, self.ref_dirs, self.pop_size)
        self.pop = pop[I]

        self.Ra = 1 - self.n_gen / self.termination.n_max_gen

    def _set_optimum(self, **kwargs):
        self.opt = filter_optimum(self.archive, least_infeasible=True)


parse_doc_string(ICMA.__init__)
