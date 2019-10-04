from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm


class MemeticAlgorithm(GeneticAlgorithm):

    def _next(self):
        # do the mating using the current population
        self.off = self._mating(self.pop)

        ################################################################
        # Add a local optimization here
        ################################################################

        # evaluate the offspring
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        # merge the offsprings with the current population
        self.pop = self.pop.merge(self.off)

        # the do survival selection
        self.pop = self.survival.do(self.problem, self.pop, self.pop_size, algorithm=self)
