import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population


class DNSGA2(NSGA2):

    def __init__(self,
                 perc_detect_change=0.1,
                 perc_diversity=0.3,
                 eps=0.0,
                 version="A",
                 **kwargs):

        super().__init__(**kwargs)
        self.perc_detect_change = perc_detect_change
        self.perc_diversity = perc_diversity
        self.eps = eps
        self.version = version

    def setup(self, problem, **kwargs):
        assert not problem.has_constraints(), "DNSGA2 only works for unconstrained problems."
        return super().setup(problem, **kwargs)

    def _infill(self):
        
        return None

    def _advance(self, **kwargs):

        pop = self.pop
        X, F = pop.get("X", "F")

        # the number of solutions to sample from the population to detect the change
        n_samples = int(np.ceil(len(pop) * self.perc_detect_change))

        # choose randomly some individuals of the current population to test if there was a change
        I = self.random_state.choice(np.arange(len(pop)), size=n_samples)
        samples = self.evaluator.eval(self.problem, Population.new(X=X[I]))

        # calculate the differences between the old and newly evaluated pop
        delta = ((samples.get("F") - F[I]) ** 2).mean()

        # if there is an average deviation bigger than eps -> we have a change detected
        change_detected = delta > self.eps

        if change_detected:

            # recreate the current population without being evaluated
            pop = Population.new(X=X)

            # find indices to be replaced (introduce diversity)
            I = np.where(self.random_state.random(len(pop)) < self.perc_diversity)[0]

            # replace with randomly sampled individuals
            if self.version == "A":
                pop[I] = self.initialization.sampling(self.problem, len(I), random_state=self.random_state)

            # replace by mutations of existing solutions (this occurs inplace)
            elif self.version == "B":
                self.mating.mutation(self.problem, pop[I])
            else:
                raise Exception(f"Unknown version of D-NSGA-II: {self.version}")

            # reevaluate because we know there was a change
            self.evaluator.eval(self.problem, pop)

            # do a survival to recreate rank and crowding of all individuals
            pop = self.survival.do(self.problem, pop, n_survive=len(pop), random_state=self.random_state)

        # create the offsprings from the current population
        off = self.mating.do(self.problem, pop, self.n_offsprings, algorithm=self, random_state=self.random_state)
        self.evaluator.eval(self.problem, off)

        # merge the parent population and offsprings
        pop = Population.merge(pop, off)

        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self, random_state=self.random_state)
