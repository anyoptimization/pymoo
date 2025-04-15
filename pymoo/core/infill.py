from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair


class InfillCriterion:

    def __init__(self,
                 repair=None,
                 eliminate_duplicates=None,
                 n_max_iterations=100,
                 **kwargs):

        super().__init__()
        self.n_max_iterations = n_max_iterations
        self.eliminate_duplicates = eliminate_duplicates if eliminate_duplicates is not None else NoDuplicateElimination()
        self.repair = repair if repair is not None else NoRepair()

    def __call__(self, problem, pop, n_offsprings, **kwargs):
        return self.do(problem, pop, n_offsprings, **kwargs)

    def do(self, problem, pop, n_offsprings, **kwargs):
        n_max_iterations = kwargs.get("n_max_iterations", self.n_max_iterations)

        # the population object to be used
        off = Population.create()

        # infill counter - counts how often the mating needs to be done to fill up n_offsprings
        n_infills = 0

        # iterate until enough offsprings are created
        while len(off) < n_offsprings:

            # how many offsprings are remaining to be created
            n_remaining = n_offsprings - len(off)

            # do the mating
            _off = self._do(problem, pop, n_remaining, **kwargs)

            # repair the individuals if necessary - disabled if repair is NoRepair
            _off = self.repair(problem, _off, **kwargs)

            # eliminate the duplicates
            _off = self.eliminate_duplicates.do(_off, pop, off)

            # if more offsprings than necessary - truncate them randomly
            if len(off) + len(_off) > n_offsprings:

                # IMPORTANT: Interestingly, this makes a difference in performance for some algorithms
                n_remaining = n_offsprings - len(off)
                _off = _off[:n_remaining]

            # add to the offsprings and increase the mating counter
            off = Population.merge(off, _off)
            n_infills += 1

            # if no new offsprings can be generated within a pre-specified number of generations
            if n_infills >= n_max_iterations:
                break

        return off

    def _do(self, problem, pop, n_offsprings, **kwargs):
        pass
