import math

import numpy as np

from pymoo.core.infill import InfillCriterion
from pymoo.operators.param_control import NoParameterControl


# ---------------------------------------------------------------------------------------------------------
# Mating
# ---------------------------------------------------------------------------------------------------------


class Mating(InfillCriterion):

    def __init__(self,
                 selection,
                 crossover,
                 mutation,
                 control=NoParameterControl,
                 **kwargs):
        super().__init__(**kwargs)
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.control = control(self)

    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):

        # how many parents need to be select for the mating - depending on number of offsprings remaining
        n_matings = math.ceil(n_offsprings / self.crossover.n_offsprings)

        # do the parameter control for the mating
        control = self.control
        control.tell(pop=pop)
        control.do(n_matings)

        # if the parents for the mating are not provided directly - usually selection will be used
        if parents is None:

            # select the parents for the mating - just an index array
            parents = self.selection.do(problem, pop, n_matings, n_parents=self.crossover.n_parents, **kwargs)

        # do the crossover using the parents index and the population - additional data provided if necessary
        off = self.crossover.do(problem, parents, **kwargs)

        # now we have to consider during parameter control that a crossover can produce multiple offsprings
        for name, param in control.params.items():
            param.set(np.repeat(param.get(), self.crossover.n_offsprings))

        # do the mutation on the offsprings created through crossover
        off = self.mutation.do(problem, off, **kwargs)

        # finally attach the parameters back to the offsprings
        control.advance(off)

        return off
