import math

from pymoo.core.infill import InfillCriterion


class Mating(InfillCriterion):

    def __init__(self,
                 selection,
                 crossover,
                 mutation,
                 **kwargs):

        super().__init__(**kwargs)
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation

    def _do(self, problem, pop, n_offsprings, parents=None, random_state=None, **kwargs):

        # how many parents need to be select for the mating - depending on number of offsprings remaining
        n_matings = math.ceil(n_offsprings / self.crossover.n_offsprings)

        # if the parents for the mating are not provided directly - usually selection will be used
        if parents is None:

            # select the parents for the mating - just an index array
            parents = self.selection(problem, pop, n_matings, n_parents=self.crossover.n_parents, random_state=random_state, **kwargs)

        # do the crossover using the parents index and the population - additional data provided if necessary
        off = self.crossover(problem, parents, random_state=random_state, **kwargs)

        # do the mutation on the offsprings created through crossover
        off = self.mutation(problem, off, random_state=random_state, **kwargs)

        return off



