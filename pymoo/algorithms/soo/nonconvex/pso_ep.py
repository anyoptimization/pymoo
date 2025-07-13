"""

Particle Swarm Optimization (PSO)

-------------------------------- Description -------------------------------



-------------------------------- References --------------------------------

[1] J. Blank and K. Deb, pymoo: Multi-Objective Optimization in Python, in IEEE Access,
vol. 8, pp. 89497-89509, 2020, DOI: 10.1109/ACCESS.2020.2990567

-------------------------------- License -----------------------------------


----------------------------------------------------------------------------
"""

import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.infill import InfillCriterion
from pymoo.core.population import Population
from pymoo.core.replacement import ImprovementReplacement, is_better
from pymoo.core.variable import Real, Choice, get
from pymoo.docs import parse_doc_string
from pymoo.operators.control import EvolutionaryParameterControl
from pymoo.operators.repair.bounds_repair import repair_random_init, repair_clamp
from pymoo.operators.sampling.rnd import FloatRandomSampling, random
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.util import default_random_state
from pymoo.util.sliding_window import SlidingWindow


# =========================================================================================================
# Mating
# =========================================================================================================


@default_random_state
def pso_canonical(V, X, P_X, L_X, w, c1, c2, random_state=None):
    n_particles, n_var = X.shape
    r1, r2 = random_state.random((n_particles, n_var)), random_state.random((n_particles, n_var))
    Vp = w * V + c1 * r1 * (P_X - X) + c2 * r2 * (L_X - X)
    return Vp


@default_random_state
def pso_rotation_invariant(V, X, P_X, L_X, inertia, c1, c2, random_state=None):
    n_particles, n_var = X.shape

    r1 = random_state.random((n_particles, n_var))
    p = X + c1 * r1 * (P_X - X)

    r2 = random_state.random((n_particles, n_var))
    l = X + c2 * r2 * (L_X - X)

    G = (X + p + l) / 3
    r = np.linalg.norm(G - X, axis=1, keepdims=True)

    Vp = inertia * V + alea_sphere(G, r) - X

    return Vp


@default_random_state
def alea_sphere(G, radius, random_state=None):
    n, m = G.shape

    x = random_state.normal(size=(n, m))
    l = np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))

    r = random_state.random(size=(n, 1))
    x = r * radius * x / l
    return x + G


class Swarm(InfillCriterion):

    def __init__(self,
                 w=0.7,
                 c1=1.4,
                 c2=1.4,
                 V_max=0.2,
                 prob_mut=0.33,
                 control=EvolutionaryParameterControl,
                 **kwargs):

        super().__init__(**kwargs)
        self.w = Real(w, bounds=(0.7, 0.9), strict=(0.0, 1.0))
        self.c1 = Real(c1, bounds=(1.2, 1.6), strict=(0.0, None))
        self.c2 = Real(c2, bounds=(1.2, 1.6), strict=(0.0, None))
        self.V_max = V_max
        self.prob_mut = prob_mut

        # of parameter control should be applied on the mating level
        self.control = control(self)

    def do(self, problem, pop, n_offsprings, algorithm=None, random_state=None, **kwargs):
        control = self.control

        # let the parameter control now some information
        control.tell(pop=pop)

        # set the controlled parameter for the desired number of offsprings
        control.do(n_offsprings, random_state=random_state)

        # get the parameters that will be used
        w, c1, c2 = get(self.w, self.c1, self.c2, size=(len(pop), 1))

        # get all the population that play a role for the mating
        swarm, pbest, lbest = algorithm.swarm, algorithm.pbest, algorithm.lbest

        V, X, P_X, L_X = swarm.get("V"), swarm.get("X"), pbest.get("X"), lbest.get("X")

        Vp = pso_canonical(V, X, P_X, L_X, w, c1, c2, random_state=random_state)
        # Vp = pso_rotation_invariant(V, X, P_X, L_X, w, c1, c2, random_state=random_state)

        # if a maximum velocity has been defined
        V_max = self.V_max
        if V_max is not None:
            xl, xu = problem.bounds()
            Vp = repair_clamp(Vp, -V_max * (xu - xl), V_max * (xu - xl))

        # the position of the new swarm particles
        Xp = X + Vp

        # if adding the velocity has brought them out of bounds -> bring them back
        if problem.has_bounds():
            Xp = repair_random_init(Xp, X, *problem.bounds(), random_state=random_state)

        # do a mutation  of the global best solution (helps to keep some diversity)
        # Xm = PM(prob=1.0, eta=20).do(problem, swarm).get("X")
        # mut = pbest.get("rank") == 0
        # Xp[mut] = Xm[mut]

        # recalculate the velocity after the repair has happened
        Vp = Xp - X

        # create the population
        off = Population.new(X=Xp, V=Vp)

        # do the reset of particles if their personal bests have not moved much
        # for k, ind in enumerate(pbest):
        #     delta = algorithm.delta[k]
        #
        #     if k != algorithm.best and delta.is_full() and np.array(delta).mean() < 0.001:
        #         particle = FloatRandomSampling().do(problem, 1)[0]
        #         particle.set("V", np.zeros(problem.n_var))
        #         off[k], pbest[k], lbest[k] = particle, particle, particle
        #         delta.clear()

        # repair the individuals if necessary - disabled if repair is NoRepair
        off = self.repair(problem, off, **kwargs)

        # advance the parameter control by attaching them to the offsprings
        control.advance(off)

        return off


@default_random_state
def get_neighbors(name, N, random_state=None):
    if name == "star":
        return np.tile(np.arange(N), (N, 1))
    elif name == "ring":
        return (np.array([np.arange(3) for _ in range(N)]) + np.arange(N)[:, None] - 1) % N
    elif name.startswith("random"):
        K = 3
        neighbors = []
        for i in range(N):
            vals = random_state.permutation(N)[:K]
            neighbors.append([i] + vals.tolist())
        return neighbors
    else:
        raise Exception(f"Unknown topology: {name}")


# =========================================================================================================
# Implementation
# =========================================================================================================


class EPPSO(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 swarm=Swarm(),
                 topology="star",
                 init_V="zero",
                 output=SingleObjectiveOutput(),
                 **kwargs):

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         mating=swarm,
                         init_V=init_V,
                         n_offsprings=None,
                         eliminate_duplicates=NoDuplicateElimination(),
                         output=output,
                         **kwargs)

        # how the initial weights should be created
        self.init_V = Choice(init_V, options=["zero", "random"])

        # how the individual are connected to determine the local (or also global) best
        self.topology = Choice(topology, options=["star", "ring"])

        # create the neighbors of each particle given the topology
        # Note: neighbors will be initialized in _initialize_advance to ensure proper random_state usage
        self.neighbors = None

        # choose the single-objective default termination
        self.termination = DefaultSingleObjectiveTermination()

        # the particles that fly around to find good solutions (pop is the pbest)
        self.swarm = None

        # the personal and local best solution
        self.lbest = None
        self.pbest = None

        # the integer of the currently best individual
        self.best = None

    def _initialize_infill(self):
        swarm = super()._initialize_infill()

        n_var = self.problem.n_var
        init_V = get(self.init_V)
        if init_V == "zero":
            V = np.zeros((len(swarm), n_var))
        elif init_V == "random":
            Xp = random(self.problem, len(swarm), random_state=self.random_state)
            V = (swarm.get("X") - Xp) / 2
        else:
            raise Exception("Unknown velocity initialization.")
        swarm.set("V", V)

        return swarm

    def _initialize_advance(self, infills=None, **kwargs):
        self.swarm = infills
        self.pbest = self.pop
        self.lbest = Population.create(*self.pbest)
        self.delta = [SlidingWindow(30) for _ in range(len(infills))]

        # Initialize neighbors with proper random_state
        if self.neighbors is None:
            self.neighbors = get_neighbors(get(self.topology), len(infills), random_state=self.random_state)

        FitnessSurvival().do(self.problem, self.pbest, return_indices=True)

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus 'infills' must to be provided."

        X = self.pbest.get("X")

        self.swarm = infills
        ImprovementReplacement().do(self.problem, self.pbest, infills, inplace=True)
        Xp = self.pbest.get("X")

        xl, xu = self.problem.bounds()
        delta = np.max(np.abs(X - Xp) / (xu - xl), axis=1)
        [self.delta[k].append(delta[k]) for k in range(len(delta))]

        pbest = self.pbest
        S = FitnessSurvival().do(self.problem, pbest, return_indices=True)
        rank = pbest.get("rank")
        self.best = S[0]

        if get(self.topology) == "random-adaptive" and pbest[self.best].get("n_gen") != self.n_gen:
            self.neighbors = get_neighbors(get(self.topology), len(pbest), random_state=self.random_state)

        # send the message from each particle to all its neighbors
        msgs = [[] for _ in range(len(pbest))]
        for k, neighbors in enumerate(self.neighbors):
            for neighbor in neighbors:
                msgs[neighbor].append(k)

        # now receive the messages and set the new local best (if an improvement has been found)
        for k, msg in enumerate(msgs):

            # if messages have been received
            if len(msg) > 0:

                # find the best one from the swarm that have been send
                i = msg[rank[msg].argmin()]

                # if the best from the message is better than the current local best
                if is_better(pbest[i], self.lbest[k]):
                    self.lbest[k] = pbest[i]

        self.pop = self.pbest

    def _set_optimum(self, **kwargs):
        k = self.pop.get("rank") == 0
        self.opt = self.pop[k]


parse_doc_string(EPPSO.__init__)

