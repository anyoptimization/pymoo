import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.problems.dyn import DynamicProblem


class DNSGA2(NSGA2):

    def __init__(self,
                 perc_sample=0.1,
                 zeta=0.3,
                 eps=0.0,
                 **kwargs):

        super().__init__(**kwargs)
        self.perc_sample = perc_sample
        self.zeta = zeta
        self.eps = eps

    def setup(self, problem, **kwargs):
        assert not problem.has_constraints(), "DNSGA2 only works for unconstrained problems."
        return super().setup(problem, **kwargs)

    def _advance(self, **kwargs):

        # for dynamic problems without real-time iterate to the next time step
        if isinstance(self.problem, DynamicProblem):
            self.problem.next()

        pop = self.pop
        X, F = pop.get("X", "F")

        # the number of solutions to sample from the population to detect the change
        n_samples = int(np.ceil(len(pop) * self.perc_sample))

        # choose randomly some individuals of the current population to test if there was a change
        I = np.random.choice(np.arange(len(pop)), size=n_samples)
        samples = self.evaluator.eval(self.problem, Population.new(X=X[I]))

        # calculate the differences between the old and newly evaluated pop
        delta = ((samples.get("F") - F[I]) ** 2).mean()

        # if there is an average deviation bigger than eps -> we have a change detected
        change_detected = delta > self.eps

        if change_detected:
            # just for now actually print there was a change
            print(self.n_gen, "CHANGE")

            # recreate the current population without being evaluated
            pop = Population.new(X=X)

            # randomly replace zeta percent to introduce some diversity
            rnd_repl = np.where(np.random.random(len(pop)) < self.zeta)[0]
            pop[rnd_repl] = self.initialization.sampling.do(self.problem, len(rnd_repl))

            # reevaluate because we know there was a change
            self.evaluator.eval(self.problem, pop)

            # do a survival to recreate rank and crowding of all individuals
            self.survival.do(self.problem, pop, n_survive=len(pop))

        # create the offsprings from the current population
        off = self.mating.do(self.problem, pop, self.n_offsprings, algorithm=self)
        self.evaluator.eval(self.problem, off)

        # merge the parent population and offsprings
        pop = Population.merge(pop, off)

        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self)
