import math

import numpy as np


class Mating:

    def __init__(self,
                 selection,
                 crossover,
                 mutation,
                 repair=None,
                 eliminate_duplicates=None,
                 n_max_iterations=100):

        super().__init__()
        self.n_max_iterations = n_max_iterations
        self.eliminate_duplicates = eliminate_duplicates
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.repair = repair

    def do(self, problem, pop, n_offsprings, **kwargs):

        # the population object to be used
        off = pop.new()

        # mating counter - counts how often the mating needs to be done to fill up n_offsprings
        n_matings = 0

        # iterate until enough offsprings are created
        while len(off) < n_offsprings:

            # how many offsprings are remaining to be created
            n_remaining = n_offsprings - len(off)

            # do the mating
            _off = self._do(problem, pop, n_remaining, **kwargs)

            if self.eliminate_duplicates is not None:
                _off = self.eliminate_duplicates.do(_off, pop, off)

            # if more offsprings than necessary - truncate them randomly
            if len(off) + len(_off) > n_offsprings:
                n_remaining = n_offsprings - len(off)
                I = np.random.permutation(len(_off))[:n_remaining]
                _off = _off[I]

            # add to the offsprings and increase the mating counter
            off = off.merge(_off)
            n_matings += 1

            # if no new offsprings can be generated within a pre-specified number of generations
            if n_matings > self.n_max_iterations:
                break

        return off

    def _do(self, problem, pop, n_offsprings, **kwargs):

        # how many parents need to be select for the mating - depending on number of offsprings remaining
        n_select = math.ceil(n_offsprings / self.crossover.n_offsprings)

        # select the parents for the mating - just an index array
        parents = self.selection.do(pop, n_select, self.crossover.n_parents, **kwargs)

        # do the crossover using the parents index and the population - additional data provided if necessary
        _off = self.crossover.do(problem, pop, parents, **kwargs)

        # do the mutation on the offsprings created through crossover
        _off = self.mutation.do(problem, _off, **kwargs)

        # repair the individuals if necessary
        if self.repair:
            _off = self.repair.do(problem, _off, **kwargs)

        return _off
